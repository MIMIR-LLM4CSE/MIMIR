from __future__ import annotations

import logging
import os
import re
import json
from contextlib import AsyncExitStack
from typing import Any

import ollama

from mcp import ClientSession

from .guardrails.policy.approval import ApprovalManager, denial_kind
from .config import (
    DEFAULT_MODEL,
    DEFAULT_THINKING_DEPTH,
    THINKING_DEPTH_BUDGETS,
    clamp_thinking_depth,
    MAX_AGENT_STEPS,
    STATE_DIR,
    SERVER_BASE,
    SKILL_BASE,
    SERVERS,
    VALID_MODES,
)
from .config.models import enforcement_level
from .context import (
    CARRY,
    SOURCE_FILE_EXTENSIONS,
    build_execution_context,
    carry_context_from_json,
    carry_context_to_json,
    carry_path_fields,
    fields_with,
    validate_execution_context,
)
from .context.capabilities import (
    DELEGATE,
    NON_BATCH,
    SENSITIVE,
    TASK_PLANNING,
    ToolCaps,
    fallbacks,
    is_write,
    names_with_cap,
    path_args,
    unannotated_live_tools,
)
from .prompt.system_prompt import (
    build_base_system_content,
    build_system_content,
)
from .tool_execution.validation import (
    absolute_workspace_path,
    auto_validate_written_file,
)
from .integration.server_manager import connect_server as connect_server_runtime
from .query_engine import run_agent_query
from .event_sink import set_event_sink, reset_event_sink
from .tool_execution.formatter import (
    json_error_payload,
    normalize_arguments,
    normalize_tool_content,
    parse_tool_payload,
    truncate_text,
)
from .tool_execution.normalizer import (
    normalize_tool_arguments,
    normalize_tool_path_argument,
    normalize_workspace_path,
    parent_path,
    rewrite_tool_for_context,
)
from .tool_execution.executor import execute_tool_call


logger = logging.getLogger(__name__)

_BASE = SERVER_BASE

_FRONT_MATTER_RE = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n(.*)$",
    re.DOTALL,
)

def _parse_skill_markdown(text: str) -> dict:
        """
        Parse SKILL.md with YAML-like front-matter.
        Returns: {name, description, content}
        """
        
        m = _FRONT_MATTER_RE.match(text)
        if not m:
            raise ValueError("SKILL.md is missing front-matter block")

        header, body = m.groups()

        meta: dict[str, str] = {}
        for line in header.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()

        if "name" not in meta or "description" not in meta:
            raise ValueError("SKILL.md front-matter must define name and description")

        return {
            "name": meta["name"],
            "description": meta["description"],
            "content": body.strip(),
        }


def _normalize_msg(msg):
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    return vars(msg)


# Execution-context fields MIRRORED into carry_context and back — derived from the
# CARRY trait declared next to each field, not hand-listed.
#
# Deliberately narrower than ``carry_set_fields()``: that one is every set-valued key
# carry_context holds, including those with no execution-context counterpart
# (``last_query_written_files``), and it answers a different question — what must be
# converted for JSON. Merging is only ever about the mirrored subset.
def _mirrored_carry_fields() -> tuple[str, ...]:
    return fields_with(CARRY)


class MimirAgent:
    """Multi-server MCP agent backed by a local Ollama model."""

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        import os
        # Publish the active model to the environment so sub-agents (spawned via
        # server_spawn_agent.py in a separate thread) inherit the same model.
        if model:
            os.environ["MIMIR_DEFAULT_MODEL"] = model
        # Resolve the scratchpad home once, here, and publish it: the ownership check
        # on a world-writable /tmp must happen in exactly one place, and both this
        # process and the server subprocesses then read the same answer from the
        # environment instead of re-deriving (and re-vetting) the path per call.
        # A refused /tmp (symlink, foreign owner) falls back to the private state dir,
        # which is where the scratchpad used to live — degraded, never absent.
        from ..servers._shared.state_paths import ensure_scratch_home, scratch_home
        resolved = ensure_scratch_home()
        if not resolved:
            resolved = os.path.join(STATE_DIR, "scratch")
            logger.warning(
                "scratchpad under %s is not usable (not ours, or not a directory); "
                "falling back to %s", scratch_home(), resolved,
            )
        os.environ["MIMIR_SCRATCH_DIR"] = resolved
        self.backend: str = os.environ.get("LLM_BACKEND", "vllm")
        self.exit_stack = AsyncExitStack()
        self.mode = "agent"
        # Reasoning depth: a single rung on the THINKING_DEPTH_* ladder is the source
        # of truth; `thinking` and `thinking_budget` are derived from it (and pushed
        # to the backend as enable_thinking / thinking_budget). Defaults to "auto" —
        # thinking on, unbudgeted, with the calibration directive in the system prompt
        # so the model spends reasoning in proportion to the difficulty of the turn.
        self.thinking_depth: int = DEFAULT_THINKING_DEPTH
        self.thinking = self.thinking_depth > 0
        self.thinking_budget: int = THINKING_DEPTH_BUDGETS[self.thinking_depth]  # -1 = unlimited; >0 = token budget hint
        self.streaming = True
        # How the per-step discovery pin is attached to the prompt (see
        # agent_loop._inject_pin): "system" (default, a transient tail system
        # message — matches the existing skill-context message), "user", or
        # "append_user" for strict templates (Mistral/Devstral). Overridable per
        # model via the vLLM profile's "pin_role" or MIMIR_PIN_ROLE. Empty means
        # "auto": resolve_pin_role() then derives it from the model profile.
        self.pin_role: str = os.environ.get("MIMIR_PIN_ROLE", "")
        # Reasoning-babysitting (guidance) enforcement level: "strict" | "light" |
        # "off". Resolved once here from the model's vLLM profile — the model is
        # immutable for the agent's lifetime, so there is nothing to re-resolve per
        # turn. Overridable at runtime via set_enforcement / the /enforcement command.
        self.enforcement: str = enforcement_level(model)
        # When True, an interactive front-end (CLI / WebSocket) has wired up a
        # continue-prompt handler, so the agent loop may ask the user to extend a
        # long run past the soft step budget. Off by default (sub-agents/tests).
        self.allow_continue_prompt = False

        self.sessions: dict[str, ClientSession] = {}
        self.tool_owner: dict[str, str] = {}
        self.tools: list[dict] = []
        # uri (str) -> {"name", "description", "mimeType", "session"}: MCP resources
        # discovered from each server at connect time. Unlike tools, resources are
        # user-attached context (Claude/Copilot-style @-mention), never model-called.
        # See context/resource_context.py and read_resource().
        self.resources: dict[str, dict] = {}
        # name -> ToolCaps: tool semantics derived from each server's declarations
        # at connect time (see context/capabilities.py). The authoritative source
        # for classification that the policy/approval/execution layers consult.
        self.tool_caps: dict[str, ToolCaps] = {}
        # Use default non_batch_tools for immediate approval of execution/compilation tools
        # Constructed empty; classification is seeded from the live per-agent
        # registry after servers connect (seed_classification_from_caps).
        self.approvals = ApprovalManager()
        self.session_approved_scopes = self.approvals.approved_scopes

        self.plan_todos: list[str] = []
        self.last_plan_query: str = ""
        # Fields carried forward from previous queries within this session.
        self._carry_context: dict[str, Any] = {}
        # Per-query read-only tool result cache; reset at the start of each query.
        self._tool_cache: dict = {}
        # Full message list from the last completed query (system message excluded).
        # Used by chat_session in full-context mode to keep tool results in history.
        self._last_full_messages: list[dict] = []
        # Reference to the message list of the query currently running (system message
        # INCLUDED, index 0). Front-ends read it to report context usage mid-turn.
        self._live_messages: list[dict] | None = None
        # Context mode: "compact" (default, aggressive compaction) or
        # "full" (keep all tool messages in history — requires large context model).
        self.context_mode: str = "full"
        # Cancellation flag: set by ws_server when the user presses Stop.
        # _stream_chat checks this on every LLM chunk so streaming aborts immediately.
        import threading as _threading
        self._cancel_flag = _threading.Event()

        self.skills: dict[str, dict[str, str]] = {}
        self.load_skills(SKILL_BASE)
        # Layer user-provided skills from .mimir/skills on top (a same-named user skill
        # overrides the bundled one). See extensions.resolve_skills_dir.
        from .extensions import resolve_skills_dir
        self.load_skills(resolve_skills_dir(), merge=True)
        self.classifier_model: str | None = None  # optionnel

        # Application extension packs: import custom policy/nudge modules from the
        # plugins dir so their descriptors register before the first query. Locked
        # policies + toggleable nudges; see client/extensions/plugins.py.
        try:
            from .extensions import load_plugins
            load_plugins()
        except Exception:
            pass

        # Operator toggles (soft-hide): servers/skills/nudges the user switched off in
        # the panel. Only *disabled* names are persisted (.mimir/preferences.json);
        # anything absent is enabled, so new servers/skills/nudges default to on. Disabled
        # servers keep their subprocess but their tools are not advertised to the LLM;
        # disabled skills are excluded from auto-detection; disabled nudges are skipped by
        # the nudge dispatcher. Application policies are locked (no toggle). See
        # config/preferences.py.
        from .config.preferences import load_disabled
        self.disabled_servers, self.disabled_skills, self.disabled_nudges = load_disabled()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = (mode or "").strip().lower()
        if normalized in VALID_MODES:
            return normalized
        raise ValueError(
            "Invalid mode. Use " + " or ".join(f"'{m}'" for m in VALID_MODES) + "."
        )

    def set_mode(self, mode: str) -> None:
        self.mode = self._normalize_mode(mode)

    def set_thinking_depth(self, level: int) -> None:
        """Move to a rung of the thinking ladder, keeping the derived flags coherent."""
        self.thinking_depth = clamp_thinking_depth(level)
        self.thinking = self.thinking_depth > 0
        self.thinking_budget = THINKING_DEPTH_BUDGETS[self.thinking_depth]

    def set_thinking(self, enabled: bool) -> None:
        """Legacy on/off switch: "on" now means "auto" (the model calibrates)."""
        self.set_thinking_depth(DEFAULT_THINKING_DEPTH if enabled else 0)

    def set_thinking_budget(self, budget: int) -> None:
        """Override the token budget without moving the rung (manual/advanced use)."""
        self.thinking_budget = budget

    def set_streaming(self, enabled: bool) -> None:
        self.streaming = enabled

    def set_backend(self, mode: str) -> None:
        import os
        from .query_engine.backends.factory import clear_backend_cache
        os.environ["LLM_BACKEND"] = mode
        self.backend = mode
        clear_backend_cache()

    def set_batch_mode(self, enabled: bool) -> None:
        self.approvals.batch_mode = enabled

    def set_context_mode(self, mode: str) -> None:
        normalized = (mode or "").strip().lower()
        if normalized not in ("compact", "full"):
            raise ValueError("Invalid context mode. Use 'compact' or 'full'.")
        self.context_mode = normalized

    def set_enforcement(self, level: str) -> None:
        normalized = (level or "").strip().lower()
        if normalized not in ("strict", "light", "off"):
            raise ValueError("Invalid enforcement level. Use 'strict', 'light', or 'off'.")
        self.enforcement = normalized

    @staticmethod
    def _normalize_arguments(arguments: Any) -> dict[str, Any]:
        return normalize_arguments(arguments)

    @staticmethod
    def _normalize_tool_content(result: Any) -> str:
        return normalize_tool_content(result)

    @staticmethod
    def _truncate_text(text: str, limit: int = 600) -> str:
        return truncate_text(text, limit=limit)

    @staticmethod
    def _json_error_payload(message: str, hint: str = "", **extra) -> str:
        return json_error_payload(message, hint=hint, **extra)

    @staticmethod
    def _parse_tool_payload(result_text: str) -> dict[str, Any] | None:
        return parse_tool_payload(result_text)

    @staticmethod
    def _normalize_workspace_path(path: str | None) -> str:
        return normalize_workspace_path(path)

    @staticmethod
    def _normalize_tool_path_argument(path: str | None) -> str:
        return normalize_tool_path_argument(path)

    def _normalize_tool_arguments(self, tool_name: str, arguments: dict) -> dict:
        path_arg_names = path_args(tool_name, self.tool_caps)
        return normalize_tool_arguments(
            tool_name,
            arguments,
            path_args_by_tool={tool_name: path_arg_names} if path_arg_names else {},
            normalize_tool_path_argument_fn=self._normalize_tool_path_argument,
        )

    def _rewrite_tool_for_context(self, tool_name: str, arguments: dict) -> tuple[str, dict]:
        return rewrite_tool_for_context(
            tool_name,
            arguments,
            tool_owner=self.tool_owner,
            is_code_filepath=self._is_code_filepath,
            normalize_workspace_path_fn=self._normalize_workspace_path,
        )

    @staticmethod
    def _parent_path(path: str) -> str:
        return parent_path(path)

    @staticmethod
    def _new_execution_context() -> dict[str, Any]:
        context = build_execution_context()
        validate_execution_context(context)
        return context

    @staticmethod
    def _is_code_filepath(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in SOURCE_FILE_EXTENSIONS

    def approval_scope(self, tool_name: str, arguments: dict) -> str:
        """The normalised approval scope for this call ("" if it cannot be derived).

        The same token the "always" grants are keyed on, reused here as the identity
        of the *goal* a refusal is about: `pip install numpy` and `pip install scipy`
        share a scope, so refusing one escalates the other, while an unrelated command
        family is untouched.
        """
        try:
            return self.approvals._approval_scope(
                tool_name, self.tool_owner.get(tool_name, "unknown"), arguments
            )
        except Exception:
            return ""

    def _record_denied_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        execution_context: dict | None,
        note: str = "",
    ) -> None:
        """Record a refusal in both ledgers.

        ``denied_tool_calls`` is the *open* set — it is cleared once the same action
        later succeeds, and drives the completion report. ``denial_history`` is
        append-only and keyed by approval scope: it is what the escalation ladder
        counts, and it must survive that clearing or a refusal followed by a success
        elsewhere would silently reset the ladder.
        """
        if execution_context is None:
            return
        fallback_tools = list(self.approvals.fallback_suggestions(tool_name))
        scope = self.approval_scope(tool_name, arguments)
        kind = denial_kind(note)
        execution_context.setdefault("denied_tool_calls", []).append({
            "tool": tool_name,
            "path": self._normalize_workspace_path(arguments.get("path") or arguments.get("filepath")),
            "fallback_tools": fallback_tools,
            "scope": scope,
            "kind": kind,
            "reason": note,
        })
        execution_context.setdefault("denial_history", []).append({
            "tool": tool_name,
            "scope": scope,
            "kind": kind,
            "reason": note,
        })

    async def _auto_validate_written_file(self, path: str, execution_context: dict | None) -> str:
        return await auto_validate_written_file(
            path=path,
            execution_context=execution_context,
            tool_owner=self.tool_owner,
            run_tool=lambda tool, args, context: self._run_tool(tool, args, execution_context=context),
            is_code_filepath=self._is_code_filepath,
            absolute_workspace_path_fn=absolute_workspace_path,
        )

    # Canonical argument key names that carry a target file path.
    _PATH_ARG_KEYS: tuple[str, ...] = ("path", "filepath", "filePath", "file_path", "file")

    def _is_write_tool(self, tool_name: str) -> bool:
        """Kept for backward compatibility; prefer get_tool_file_targets."""
        return is_write(tool_name, self.tool_caps)


    def get_tool_file_targets(self, tool_name: str, arguments: dict) -> list[str]:
            """
            Return the list of normalized workspace paths that this tool call
            may mutate.  We snapshot them before execution and compare after;
            tools that turn out to be read-only produce no diff.
            """

            # Generic: any tool that carries a file-path argument.
            # We snapshot the file before execution and compare after — tools
            # that don't modify it (reads, checks, etc.) produce no diff.
            for key in self._PATH_ARG_KEYS:
                raw = arguments.get(key)
                if isinstance(raw, str) and raw:
                    normalized = self._normalize_workspace_path(raw)
                    if normalized:
                        return [normalized]

            return []


    def _get_todo_file(self) -> str:
        """Return the absolute path to the active todo_list.md, or '' if not available."""
        if not names_with_cap(TASK_PLANNING, self.tool_caps):
            return ""
        mimir_dir = STATE_DIR
        sidecar = os.path.join(mimir_dir, "active_session")
        todo_file = ""
        try:
            if os.path.exists(sidecar):
                with open(sidecar, "r", encoding="utf-8") as _f:
                    _sid = _f.read().strip()
                if _sid:
                    todo_file = os.path.join(mimir_dir, "sessions", _sid, "todo_list.md")
        except OSError:
            pass
        if not todo_file:
            todo_file = os.path.join(mimir_dir, "todo_list.md")
        return todo_file

    async def _build_system_content(self, active_mode: str) -> str:
        memory_file = os.path.join(STATE_DIR, "memory", "MEMORY.md") if "memory_search" in self.tool_owner else ""
        todo_file = self._get_todo_file()

        return build_system_content(
            active_mode=active_mode,
            tool_owner=self.tool_owner,
            sensitive_tools=self.approvals.sensitive_tools,
            memory_context_file=memory_file,
            todo_file=todo_file,
            plan_todos=self.plan_todos,
            thinking_depth=self.thinking_depth,
            delegation_available=bool(names_with_cap(DELEGATE, self.tool_caps)),
        )

    def _apply_carry_context(self, execution_context: dict) -> None:
        """Merge long-lived session knowledge into a fresh per-query execution context.

        Only accumulative, additive fields are carried forward.  Per-query state
        (workflow phases, nudge counts, edit signatures, denied calls) is intentionally
        reset so each query starts clean from a policy perspective.

        Files in read_files whose on-disk mtime has changed since they were read are
        evicted so the policy does not falsely treat them as already-confirmed reads.
        """
        # Evict stale read_files entries before merging.
        prior_reads: set[str] = self._carry_context.get("read_files", set())
        if prior_reads:
            read_mtimes: dict[str, float] = self._carry_context.get("_read_mtimes", {})
            fresh_reads: set[str] = set()
            for p in prior_reads:
                recorded_mtime = read_mtimes.get(p)
                if recorded_mtime is None:
                    fresh_reads.add(p)  # no mtime recorded — keep as-is
                    continue
                try:
                    current_mtime = os.path.getmtime(absolute_workspace_path(p))
                    if current_mtime <= recorded_mtime:
                        fresh_reads.add(p)
                    # else: file modified on disk — drop from carry
                except OSError:
                    pass  # file deleted; drop it
            self._carry_context["read_files"] = fresh_reads

        for field in _mirrored_carry_fields():
            prior: set = self._carry_context.get(field, set())
            if isinstance(prior, set) and isinstance(execution_context.get(field), set):
                # sorted(): carried entries seed a RecencySet's order, and a plain set
                # iterates in hash order — which varies per process. Insert them in a
                # stable order so the pin renders identically across runs. They are all
                # older than anything this query records, so relative order among them
                # carries no recency information to preserve.
                execution_context[field].update(sorted(prior))
        if self._carry_context.get("searched"):
            execution_context["searched"] = True
        # Forward previous-query write set into execution context so the
        # discovery pin can warn the model to re-read those files.
        prev_written = self._carry_context.get("last_query_written_files", set())
        if prev_written:
            execution_context["prev_query_written_files"] = set(prev_written)

    def _update_carry_context(self, execution_context: dict) -> None:
        """Persist the fields worth remembering into the session carry dict."""
        for field in _mirrored_carry_fields():
            current: set = execution_context.get(field, set())
            if isinstance(current, set):
                prior: set = self._carry_context.get(field, set())
                self._carry_context[field] = prior | current
        if execution_context.get("searched"):
            self._carry_context["searched"] = True
        # Record which files were written this query so the next query's
        # discovery pin can warn the model to re-read them before editing.
        self._carry_context["last_query_written_files"] = set(
            execution_context.get("dirty_written_files", set())
        )
        # Evict files that were written this query from read_files carry.
        # The agent just changed them, so the old read evidence is stale — a
        # fresh read_file must happen before the next query edits them again.
        dirty: set[str] = execution_context.get("dirty_written_files", set())
        read_mtimes: dict[str, float] = self._carry_context.setdefault("_read_mtimes", {})
        if dirty:
            carry_reads: set[str] = self._carry_context.get("read_files", set())
            carry_reads -= dirty
            # Also drop their mtimes so the next apply_carry_context cannot
            # accidentally resurrect them as "up to date" entries.
            for p in dirty:
                read_mtimes.pop(p, None)

        # Stamp every carried read with the mtime observed NOW. Unconditionally: the
        # read happened during this query, so the file's current state is the baseline
        # the next query must compare against. Recording only the first time (the old
        # `if p not in read_mtimes`) froze the baseline forever, so a single external
        # edit evicted the path from carry on every subsequent query — the model re-read
        # it each time and the evidence was never accepted again.
        carried_reads: set[str] = self._carry_context.get("read_files", set())
        for p in carried_reads:
            try:
                read_mtimes[p] = os.path.getmtime(absolute_workspace_path(p))
            except OSError:
                read_mtimes.pop(p, None)
        # Drop mtimes for paths no longer carried, so the dict cannot grow unboundedly
        # across a long session.
        for p in [p for p in read_mtimes if p not in carried_reads]:
            del read_mtimes[p]

    def _discard_carry_path(self, path: str) -> None:
        """Remove a deleted file from every carried field that holds file paths.

        Derived rather than listed, and derived from the *carry* schema rather than from
        the execution-context traits alone: ``last_query_written_files`` exists only in
        carry_context, so a trait-only derivation could not see it and a deleted file
        kept being announced in the pin as "written last query — re-read before editing".
        ``inspected_dirs`` is correctly absent: it holds directories, and deleting a file
        does not un-inspect the directory that contained it.
        """
        for field in carry_path_fields():
            self._carry_context.get(field, set()).discard(path)
        self._carry_context.get("_read_mtimes", {}).pop(path, None)

    def _request_tool_approval(self, tool_name: str, arguments: dict, max_attempts: int = 3) -> tuple[bool, str]:
        return self.approvals.request(
            tool_name=tool_name,
            server_name=self.tool_owner.get(tool_name, "unknown"),
            arguments=arguments,
            max_attempts=max_attempts,
        )

    def _request_path_approval(
        self, abspath: str, tool_name: str, arguments: dict | None = None
    ) -> tuple[bool, bool]:
        """Approve out-of-workspace access to *abspath*. Returns (approved, always).

        Default terminal prompt (allow once / always / deny). *arguments* are the
        call's own arguments, shown as the usual tool description so the user sees
        what the tool is doing, not just which path it touches. Front-ends override:
        the WebSocket worker routes this through the approval UI; the headless runner
        auto-approves. Non-interactive stdin (EOF) fails closed (denied).
        """
        from . import human_pause
        from .context.capabilities import label_for

        # The header names the action with a short file name; the absolute path is
        # printed on its own line below, because *where* is the decision the user is
        # being asked to make. Shortening only the header keeps the prompt readable
        # without ever hiding the location being authorised.
        from .tool_execution.tool_status_messages import shorten_display_args

        short_args = shorten_display_args(tool_name, arguments or {}, self.tool_caps)
        what = label_for(tool_name, short_args, self.tool_caps) or tool_name
        try:
            print(f"\n⚠ {what}\n  needs access outside the workspace:\n  {abspath}",
                  flush=True)
            # Waiting on the user, not on the tool (see human_pause).
            with human_pause.human_pause():
                reply = input("  Allow once [y] / always [a] / deny [N]? ").strip().lower()
        except EOFError:
            return (False, False)
        if reply in ("a", "always"):
            return (True, True)
        if reply in ("y", "yes", "once"):
            return (True, False)
        return (False, False)

    def _request_continue(self, summary: str) -> bool:
        """Ask whether to continue past the soft step budget.

        Default (non-interactive) behaviour returns False so the run stops at the
        budget. Interactive front-ends replace this with a handler that prompts
        the user (CLI ``input()`` or a WebSocket continue card).
        """
        return False

    def _request_user_question(self, questions: list) -> dict:
        """Ask the user one or more clarifying questions (``ask_user_question`` tool).

        ``questions`` is a list of ``{header, question, multiSelect, options}`` specs
        to ask in order. Returns ``{"answers": [{"selected": [<labels>],
        "other_text": <str|None>}, ...]}`` — one entry per question. The default
        (non-interactive) behaviour returns no answers, which the elicitation callback
        maps to a ``decline`` so the model proceeds with its best judgment. Interactive
        front-ends replace this with a handler that prompts the user sequentially
        (CLI ``input()`` or a WebSocket question card).
        """
        return {"answers": []}

    def _denied_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        note: str = "",
        execution_context: dict | None = None,
    ) -> str:
        """The tool result a refused approval puts back in front of the model.

        Not "ask again": a refusal is an instruction, and the hint states the three
        readings it can carry so the model picks one instead of retrying. The stage
        (see guardrails.workflow) narrows that choice as refusals accumulate.
        """
        from .guardrails.workflow import (
            STAGE_DROP_OR_STOP,
            STAGE_HANDBACK,
            denial_scope_count,
            denial_stage,
        )

        scope = self.approval_scope(tool_name, arguments)
        context = execution_context if execution_context is not None else {}
        stage = denial_stage(context, scope)
        refusals = denial_scope_count(context, scope)

        fallback_tools = list(self.approvals.fallback_suggestions(tool_name))
        if stage == STAGE_HANDBACK:
            hint = (
                "This has now been refused repeatedly. Do not look for another route and do not "
                "ask again: stop here and hand back. Finish your turn with what you did complete, "
                "what is blocked by the refusal, and what you need from the user to continue."
            )
        elif stage == STAGE_DROP_OR_STOP:
            hint = (
                "This was refused more than once, so the objection is to the action itself, not "
                "just to how you went about it. Do not try another route to the same end. Either "
                "drop this step and continue the rest of the task without it — saying plainly that "
                "it was skipped at the user's request — or, if the task cannot go on without it, "
                "stop and hand back with what is blocked."
            )
        else:
            hint = (
                "Read the refusal before reacting; it carries one of three meanings, in this order "
                "of priority. (1) Not this way: the goal is fine but the means is wrong — reach it "
                "another way. (2) Unnecessary: the step is not needed — drop it, continue the rest "
                "of the task, and report it as skipped at the user's request. (3) Stop: end your "
                "turn and hand back with what is done, what is blocked, and what you need. "
                "Never re-issue this action and never route around the approval gate."
            )
            if fallback_tools:
                hint += " Safe alternatives available for (1): " + ", ".join(fallback_tools) + "."
            else:
                hint += " No safe alternative is configured for (1)."

        return self._json_error_payload(
            f"Execution of tool '{tool_name}' was refused by the user.",
            hint=hint,
            tool=tool_name,
            arguments=arguments,
            denial_reason=note or "denied by user",
            denial_kind=denial_kind(note),
            denial_stage=stage,
            prior_refusals_for_this_goal=refusals,
        )

    async def connect_server(self, name: str, script: str) -> None:
        await connect_server_runtime(agent=self, name=name, script=script)

    async def read_resource(self, uri: str) -> str:
        """Read an MCP resource's text content by URI (best-effort).

        Resources are user-attached context — the host reads them and injects the
        content into a turn (see context/resource_context.py), unlike tools which the
        model invokes. Dispatches to the owning session recorded in ``self.resources``.
        Returns "" for an unknown URI or on any read/transport error, and joins the
        text of all ``TextResourceContents`` parts (binary blobs are skipped). Never
        raises into the chat loop.
        """
        info = self.resources.get(uri)
        if not info:
            return ""
        session = self.sessions.get(info.get("session", ""))
        if session is None:
            return ""
        try:
            result = await session.read_resource(uri)
        except Exception:
            return ""
        parts = [
            c.text for c in getattr(result, "contents", [])
            if getattr(c, "text", None) is not None
        ]
        return "\n".join(parts)

    def seed_classification_from_caps(self) -> None:
        """Seed the approval manager from the live, per-agent tool registry.

        Call once after all servers are connected (the registry is now fully
        populated). The approval manager is constructed before any server connects,
        so its sensitive/non-batch/fallback classification has to be (re)seeded here
        from ``self.tool_caps`` — what each connected server actually declared.

        The three sets are mutated **in place** (clear + update), never reassigned:
        ``session_approved_scopes`` aliases ``self.approvals.approved_scopes`` and
        external code holds references to the manager, so the object identity of
        these attributes must be preserved.
        """
        self.approvals.sensitive_tools.clear()
        self.approvals.sensitive_tools.update(names_with_cap(SENSITIVE, self.tool_caps))

        self.approvals.non_batch_tools.clear()
        self.approvals.non_batch_tools.update(names_with_cap(NON_BATCH, self.tool_caps))

        self.approvals.fallback_tools.clear()
        self.approvals.fallback_tools.update(
            {name: fallbacks(name, self.tool_caps)
             for name in self.tool_caps if fallbacks(name, self.tool_caps)}
        )

        # Share the live registry so scope-narrowing, the confirm/URL sensitivity
        # gates, and risk notes read current declarations. ``agent.tool_caps`` is
        # mutated in place (server_manager) and never reassigned, so the reference
        # stays valid for tools that connect after this seed call.
        self.approvals.tool_caps = self.tool_caps

        self.report_capability_consistency()

    # ── server / skill toggles (soft-hide) ─────────────────────────────────────
    def advertised_tools(self) -> list[dict]:
        """The tool list shown to the LLM, with disabled servers' tools filtered out.

        Soft-hide: the disabled server's subprocess stays connected (its tools remain
        in ``self.tools``), but they are excluded here so the model can neither see nor
        call them. A tool with no known owner is kept (fail-open).
        """
        if not self.disabled_servers:
            return self.tools
        return [
            t for t in self.tools
            if self.tool_owner.get(t.get("function", {}).get("name", "")) not in self.disabled_servers
        ]

    def skill_enabled(self, name: str) -> bool:
        return name not in self.disabled_skills

    def nudge_enabled(self, name: str) -> bool:
        """True if an application nudge is active (consulted by the nudge dispatcher)."""
        return name not in self.disabled_nudges

    def set_server_enabled(self, name: str, enabled: bool) -> None:
        """Enable/disable a server's tools and persist the choice."""
        if enabled:
            self.disabled_servers.discard(name)
        else:
            self.disabled_servers.add(name)
        self._save_toggles()

    def set_skill_enabled(self, name: str, enabled: bool) -> None:
        """Enable/disable a skill for auto-detection and persist the choice."""
        if enabled:
            self.disabled_skills.discard(name)
        else:
            self.disabled_skills.add(name)
        self._save_toggles()

    def set_nudge_enabled(self, name: str, enabled: bool) -> None:
        """Enable/disable an application nudge and persist the choice."""
        if enabled:
            self.disabled_nudges.discard(name)
        else:
            self.disabled_nudges.add(name)
        self._save_toggles()

    def _save_toggles(self) -> None:
        from .config.preferences import save_disabled
        save_disabled(self.disabled_servers, self.disabled_skills, self.disabled_nudges)

    def toggles_state(self) -> dict:
        """Snapshot for the toggle panel: every server and skill with its enabled flag.

        Servers come from the merged registry (bundled ``SERVERS`` + discovered user
        servers under ``.mimir/servers``), so the panel lists them even if one failed to
        connect; skills from the loaded ``self.skills`` map (which already includes user
        skills). Each row carries a one-line ``description`` for the hover tooltip.
        """
        from .extensions import all_servers, all_server_descriptions
        server_descriptions = all_server_descriptions()
        servers = [
            {
                "name": name,
                "description": server_descriptions.get(name, ""),
                "enabled": name not in self.disabled_servers,
            }
            for name in all_servers()
        ]
        skills = [
            {
                "name": name,
                "description": meta.get("description", ""),
                "enabled": name not in self.disabled_skills,
            }
            for name, meta in sorted(self.skills.items())
        ]
        # Application nudges from registered extension packs. Locked application
        # policies are intentionally NOT listed — they cannot be toggled off.
        from .guardrails.nudges.plugins import NudgeRegistry
        nudges = [
            {
                "name": name,
                "description": "application nudge",
                "enabled": name not in self.disabled_nudges,
            }
            for name in NudgeRegistry.names()
        ]
        return {"servers": servers, "skills": skills, "nudges": nudges}

    def report_capability_consistency(self) -> None:
        """Warn about connected tools that resolved to no capabilities.

        Call once after all servers are connected. With classification now owned
        by the servers (each declares its caps via ``@mcp.tool(**tool_caps(...))``),
        a connected tool with an empty descriptor most likely *forgot to declare* —
        the silent-drift signal the registry is designed to surface. Genuinely pure
        tools (math, string ops, read-only queries) also resolve empty, so this is
        informational, not an error.
        """
        unannotated = unannotated_live_tools(self.tool_caps)
        if unannotated:
            logger.info(
                "Tool-capability registry: %d connected tool(s) declared no "
                "capabilities (pure tool, or missing tool_caps?): %s",
                len(unannotated), ", ".join(unannotated),
            )

    async def _run_tool(
        self,
        tool_name: str,
        arguments: dict,
        execution_context: dict | None = None,
        run_auto_validation: bool = True,
        call_id: str = "",
    ) -> str:
        return await execute_tool_call(
            agent=self,
            tool_name=tool_name,
            arguments=arguments,
            execution_context=execution_context,
            run_auto_validation=run_auto_validation,
            call_id=call_id,
        )

    async def compact_history(self, history: list[dict]) -> str:
        """Summarize accumulated conversation history into a single compact message.

        Calls Ollama with a structured compaction prompt and no tools.  Returns
        the summary string; the caller is responsible for mutating the history
        list in-place.  Returns an empty string when history is empty.
        """
        if not history:
            return ""
        messages: list[dict] = [
            {"role": "system", "content": build_base_system_content()},
            *history,
            {
                "role": "user",
                "content": (
                    "The above is a conversation history with a coding agent. "
                    "Produce a concise HANDOFF NOTE that a new session of the same agent "
                    "can read to continue work without re-discovering what was already done. "
                    "Cover:\n"
                    "1. Task(s) requested — one sentence each\n"
                    "2. Repository structure discovered — directories, key files, and their purpose\n"
                    "3. Files created or modified — path + one-sentence description of what the file does and how it fits in\n"
                    "4. Key decisions and their rationale (e.g. why a particular design was chosen)\n"
                    "5. What was validated and the result\n"
                    "6. What is still pending or incomplete\n\n"
                    "Rules:\n"
                    "- Be specific: always use full file paths, class names, function names\n"
                    "- Prefer structure over prose — use short lists\n"
                    "- Keep it under 800 words\n"
                    "- Do NOT repeat what can be inferred from file names alone; "
                    "explain non-obvious design choices and internal structure"
                ),
            },
        ]
        response = ollama.chat(
            model=self.model,
            messages=messages,
            tools=[],
            think=False,
            stream=False,
        )
        msg = response.get("message", {}) if isinstance(response, dict) else vars(response).get("message", {})
        if hasattr(msg, "content"):
            return msg.content or ""
        if isinstance(msg, dict):
            return msg.get("content") or ""
        return ""

    def compact_messages(self, middle: list[dict]) -> list[dict]:
        """Summarize a slice of conversation messages into a single summary message.

        Synchronous counterpart of :meth:`compact_history`, called from the agent
        loop (which runs in the worker thread) via ``_maybe_compact_intra_query``.
        Unlike ``compact_history`` this uses the *configured* backend — not a
        hard-coded Ollama client — so it works under vLLM too, and returns a list
        of messages (the summary) rather than a bare string.

        Robustness: the summarization request itself is bounded to the model's
        window (via the loop's force-fit helper applied to a throwaway copy) so it
        can never trigger the very overflow it exists to prevent. On any failure or
        empty result it returns *middle* unchanged, leaving the caller's truncation
        backstop to handle the size.
        """
        if not middle:
            return middle
        from .query_engine.backends.factory import get_backend
        from .query_engine.history import _force_fit_to_window, served_compaction_instruction

        backend = get_backend()
        tok = lambda t: backend.count_text_tokens(self.model, t)  # noqa: E731
        prompt: list[dict] = [
            {"role": "system", "content": build_base_system_content()},
            *[dict(m) for m in middle],
            {"role": "user", "content": served_compaction_instruction()},
        ]
        # Keep the summarization call itself within the window. Reserve ~25% for
        # the summary output; protect the system + instruction messages.
        window = backend.context_window(self.model)
        if window:
            _force_fit_to_window(prompt, int(window * 0.75), tok)
        try:
            msg = backend.chat(
                self.model, prompt, [], False, False, {"temperature": 0.2}
            )
        except Exception:
            return middle
        summary = (msg or {}).get("content", "") if isinstance(msg, dict) else ""
        if not summary:
            return middle
        n = len(middle) // 2
        return [{
            "role": "assistant",
            "content": (
                f"[Context summary — {n} prior exchange{'s' if n != 1 else ''} "
                f"compacted]\n\n{summary}"
            ),
        }]

    def load_skills(self, skills_dir: str, *, merge: bool = False) -> None:
        """
        Load skills from a directory.

        Expected layout:
        skills_dir/
            <skill-name>/
            SKILL.md

        ``merge=False`` (default) resets ``self.skills`` first (bundled load). ``merge=True``
        adds to the existing set so a user skills dir (``.mimir/skills``) can be layered on
        top of the bundled one; a user skill with the same name overrides the bundled one.
        """
        if not merge:
            self.skills = {}

        if not os.path.isdir(skills_dir):
            return

        for entry in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, entry)
            if not os.path.isdir(skill_path):
                continue

            md_path = os.path.join(skill_path, "SKILL.md")
            if not os.path.isfile(md_path):
                continue

            try:
                with open(md_path, "r", encoding="utf-8") as f:
                    parsed = _parse_skill_markdown(f.read())
            except Exception as exc:
                logger.warning("Failed to load skill %s: %s", entry, exc)
                continue

            name = parsed["name"]

            # Defensive: directory name and skill name must match
            if name != entry:
                logger.warning(
                    "Skill directory '%s' and skill name '%s' mismatch; skipping",
                    entry,
                    name,
                )
                continue

            if merge and name in self.skills:
                logger.info("User skill '%s' overrides the bundled skill of the same name", name)

            self.skills[name] = {
                "description": parsed["description"],
                "content": parsed["content"],
                # future-compatible slot:
                # "metadata": {...}
            }

    async def detect_skill_implicit(
        agent: Any,
        query: str,
        history: list[dict] | None = None,
    ) -> str | None:
        """
        Ask the model to classify whether a skill applies to the user query.

        ``history`` is the recent conversation (user + assistant messages).
        Including it allows multi-turn detection, e.g.:
            Q: "Can you plan implementing X?"
            A: "Here is the plan… want me to start?"
            Q: "Yes"  ← current query; skill detected from context
        Returns the skill name or None.
        """

        if not getattr(agent, "skills", None):
            return None

        # Only skills the operator left enabled are eligible for auto-detection
        # (soft-hide via the toggle panel). An explicitly-disabled skill is invisible
        # to the classifier and so can never be selected.
        enabled_skills = {
            name: data for name, data in agent.skills.items()
            if agent.skill_enabled(name)
        }
        if not enabled_skills:
            return None

        # Build a compact recent-conversation snippet (last 3 turns, user/assistant only).
        # Tool results and system messages are excluded to keep the prompt tight.
        conversation_context = ""
        if history:
            recent: list[dict] = [
                m for m in history
                if m.get("role") in ("user", "assistant")
            ][-6:]  # up to 3 user+assistant pairs
            if recent:
                lines = []
                for m in recent:
                    role_label = "User" if m["role"] == "user" else "Agent"
                    # Truncate long messages so the classifier prompt stays small.
                    text = str(m.get("content") or "").strip()
                    if len(text) > 400:
                        text = text[:400] + "…"
                    lines.append(f"{role_label}: {text}")
                conversation_context = (
                    "\n\nRecent conversation (for context):\n"
                    + "\n".join(lines)
                )

        # Build the skills index (name + description only) from enabled skills.
        skills_list = "\n".join(
            f"- {name}: {data.get('description', '').strip()}"
            for name, data in enabled_skills.items()
        )

        classifier_messages = [
            {
                "role": "system",
                "content": (
                    "You are a classifier.\n\n"
                    "Available skills:\n"
                    f"{skills_list}\n\n"
                    "Rules:\n"
                    "- Evaluate the LATEST user message in light of the full conversation.\n"
                    "- If the conversation context makes it clear that one skill applies "
                    "(even if the latest message alone is ambiguous, e.g. 'yes', 'go ahead'), "
                    "return that skill name.\n"
                    "- If no skill applies, return \"none\".\n"
                    "- Do not explain your reasoning.\n\n"
                    "Respond ONLY in JSON with this exact shape:\n"
                    "{\"skill\": \"<skill_name_or_none>\"}"
                    + conversation_context
                ),
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        try:
            resp = ollama.chat(
                model=agent.model,
                messages=classifier_messages,
                stream=False,
            )
        except Exception:
            return None

        resp_dict = _normalize_msg(resp)
        msg = resp_dict.get("message", {})
        if not isinstance(msg, dict):
            msg = _normalize_msg(msg)
        content = msg.get("content", "")

        try:
            parsed = json.loads(content)
        except Exception:
            return None

        skill = parsed.get("skill")
        return skill if skill in agent.skills else None

    async def run(
        self,
        query: str,
        max_steps: int = MAX_AGENT_STEPS,
        history: list[dict] | None = None,
        mode: str | None = None,
        thinking: bool = False,
        streaming: bool = True,
        token_callback: Any = None,
        think_token_callback: Any = None,
        think_start_callback: Any = None,
        think_end_callback: Any = None,
        event_callback: Any = None,
    ) -> str:
        # Bind the structured-event sink for this run so emit() in the engine and
        # tool executor (and their gathered tasks) route events to event_callback
        # instead of stdout. When no callback is supplied (CLI), the sink stays
        # unset and emit() prints — preserving the original behaviour.
        token = set_event_sink(event_callback) if event_callback else None
        try:
            return await run_agent_query(
                agent=self,
                query=query,
                max_steps=max_steps,
                history=history,
                mode=mode,
                thinking=thinking,
                streaming=streaming,
                logger=logger,
                token_callback=token_callback,
                think_token_callback=think_token_callback,
                think_start_callback=think_start_callback,
                think_end_callback=think_end_callback,
            )
        finally:
            if token is not None:
                reset_event_sink(token)

    async def chat_loop(self) -> None:
        # Lazy import: the CLI chat frontend lives in ``ui``; the core does not
        # depend on a frontend at load time.
        from .ui.cli.chat_session import run_chat_session
        await run_chat_session(self)

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()

    # ── Session state serialisation ────────────────────────────────────────────
    #
    # "Which keys must survive a JSON round-trip as sets" is answered by the carry
    # schema (``carry_set_fields``), not by the CARRY trait: the trait only knows the
    # fields mirrored from the execution context, and carry_context holds set-valued
    # keys of its own.

    def export_state(self) -> dict[str, Any]:
        """Snapshot the agent's carry_context for session persistence.

        Conversion is derived from the carry schema rather than from the CARRY trait
        alone: keys that live only in carry_context (``last_query_written_files``) are
        sets too, and the session store serialises with ``default=str``, so anything
        missed round-trips as a string instead of raising.
        """
        return {"carry_context": carry_context_to_json(self._carry_context)}

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore agent state from a previously exported snapshot."""
        self._carry_context = carry_context_from_json(state.get("carry_context", {}))


__all__ = ["MimirAgent", "SERVERS", "_BASE"]

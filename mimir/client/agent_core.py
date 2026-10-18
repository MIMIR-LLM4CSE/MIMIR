from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

from mcp import ClientSession
from pydantic import AnyUrl

from ..servers._shared.state_paths import (
    TODO_FILENAME,
    ensure_scratch_home,
    scratch_home,
    session_state_dir,
)
from .config import (
        DEFAULT_MODEL,
        DEFAULT_THINKING_DEPTH,
        GLOBAL_STATE_DIR,
        READONLY_MODES,
        SERVER_BASE,
        SERVERS,
        SKILL_BASE,
        STATE_DIR,
        THINKING_DEPTH_BUDGETS,
        WORKSPACE_ROOT,
)
from .config.models import enforcement_level
from .context.paths import absolute_workspace_path, normalize_workspace_path
from .config.preferences import load_subagent_level, load_temperature
from .context import (
        SOURCE_FILE_EXTENSIONS,
        validate_execution_context,
)
from .tools.capabilities import (
    DELEGATE,
    NON_BATCH,
    SENSITIVE,
    TASK_PLANNING,
    ToolCaps,
    fallbacks,
    is_write,
    names_with_cap,
    path_args,
    target_path_arg,
    unannotated_live_tools,
)
from .context.execution_context import ExecutionContext
from .core.human import TerminalHuman
from .core.ports import HumanIO
from .core.settings import apply_setting
from .event_sink import reset_event_sink, set_event_sink
from .core.extension_set import AMBIENT, ExtensionSet
from .core.turn_state import TurnState
from ..servers._shared.skill_file import read_skills_dir
from .config.toggles import Toggles
from .context.carry import CarryContext
from .extensions import all_server_descriptions, all_servers, load_plugins, resolve_skills_dir
from .guardrails.policy.approval import ApprovalManager, denial_kind
from .guardrails.toollist import tools_for_readonly_mode
from .guardrails.workflow import (
        STAGE_DROP_OR_STOP,
        STAGE_HANDBACK,
        denial_scope_count,
        denial_stage,
)
from .integration.server_manager import connect_server as connect_server_runtime
from .integration.server_manager import connect_servers as connect_servers_runtime
from .prompt.system_prompt import (
        build_base_system_content,
        build_system_content,
)
from .protocol.tool_result import (
        json_error_payload,
)
from .query_engine import run_agent_query
from .llm.factory import get_backend
from .query_engine.history import (
        force_fit_to_window,
        compacted_exchanges,
        compaction_summary_message,
        served_compaction_instruction,
)
from .tool_execution.executor import execute_tool_call
from .tool_execution.normalizer import (
        normalize_tool_arguments,
        normalize_tool_path_argument,
        rewrite_tool_for_context,
)
from .tool_execution.validation import auto_validate_written_file

logger = logging.getLogger(__name__)

_BASE = SERVER_BASE

def _discard_token(_text: str) -> None:
    """Token sink for model calls whose output is not the user's answer (compaction)."""


class MimirAgent:
    """Multi-server MCP agent."""

    def __init__(self, model: str = DEFAULT_MODEL, session_id: str | None = None):
        self.model = model
        self.server_env: dict[str, str] = {}
        # The tree this agent works in. The module constant for the agent the user
        # drives; a sub-agent given a copy of the repository points here instead, and
        # every root-sensitive check reads it off the agent rather than off the import.
        # Note what does NOT follow it: STATE_DIR and MIMIR_DIR stay the real
        # workspace's, so memory, preferences and .mimir extensions are shared.
        self.workspace_root: str = WORKSPACE_ROOT
        # The conversation this agent works for, for its whole life. Agents share one
        # `os.environ`, so everything session-scoped reads it from here; servers get it at spawn
        # (integration/server_manager). None for the CLI, the runner and tests.
        self.session_id: str | None = session_id
        # Neither the model nor the session goes to os.environ: they travel per server. The
        # scratchpad home is resolved and vetted once, here, then published for every process; a
        # refused /tmp falls back to the private state dir.
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
        # Reasoning-babysitting (guidance) enforcement level: "strict" | "light" |
        # "off". Resolved once here from the model's vLLM profile — the model is
        # immutable for the agent's lifetime, so there is nothing to re-resolve per
        # turn. Overridable at runtime via set_enforcement / the /enforcement command.
        self.enforcement: str = enforcement_level(model)
        # Sampling temperature the user chose for this model, or None for the model's
        # own (nothing sent). Persisted per model, so a sub-agent of the same model
        # starts from it too.
        self.temperature: float | None = load_temperature(model)
        self.sessions: dict[str, ClientSession] = {}
        self.tool_owner: dict[str, str] = {}
        self.tools: list[dict] = []
        # uri (str) -> {"name", "description", "mimeType", "session"}: MCP resources
        # discovered from each server at connect time. Unlike tools, resources are
        # user-attached context (Claude/Copilot-style @-mention), never model-called.
        # See context/resource_context.py and read_resource().
        self.resources: dict[str, dict] = {}
        # name -> ToolCaps: tool semantics derived from each server's declarations
        # at connect time (see tools/capabilities.py). The authoritative source
        # for classification that the policy/approval/execution layers consult.
        self.tool_caps: dict[str, ToolCaps] = {}
        # Use default non_batch_tools for immediate approval of execution/compilation tools
        # Constructed empty; classification is seeded from the live per-agent
        # registry after servers connect (seed_classification_from_caps).
        self.approvals = ApprovalManager(session_id=session_id)
        # Who answers for the user: the terminal, unless the front-end puts its own.
        self.human: HumanIO = TerminalHuman()
        # What a front-end lends the turn (core.ports.AgentPort); none by default.
        self.poll_steer: Callable[[], list[str]] | None = None
        self.register_background_job: Callable[..., bool] | None = None
        self.watched_background_jobs: Callable[[], list[dict]] | None = None
        self.session_approved_scopes = self.approvals.approved_scopes

        self.plan_todos: list[str] = []
        self.last_plan_query: str = ""
        # What one query leaves for the next in this session.
        self.carry = CarryContext()
        # The turn in flight and the last finished one (core.turn_state).
        self.turn = TurnState()
        # Bounds below the model's own, 0 meaning none: a sub-agent is held to a smaller
        # window and a shorter answer than the model allows.
        self.max_context_tokens: int = 0
        self.max_answer_tokens: int = 0
        # Context mode: "full" (the default — keep every tool message in history) or
        # "compact" (aggressive compaction, for a small context window).
        self.context_mode: str = "full"

        self.skills: dict[str, dict[str, Any]] = {}
        self.load_skills(SKILL_BASE)
        # Layer user-provided skills from .mimir/skills on top (a same-named user skill
        # overrides the bundled one). See extensions.resolve_skills_dir.
        self.load_skills(resolve_skills_dir(), merge=True)

        # The workspace's extension packs: locked policies, toggleable nudges, post-tool
        # hooks. A failure to load them is logged, never fatal.
        try:
            self.extensions = load_plugins()
        except Exception:
            logger.warning("extension packs could not all be loaded", exc_info=True)
            self.extensions = ExtensionSet(parent=AMBIENT)

        # The servers, skills and nudges the user switched off (config.toggles).
        self.toggles = Toggles()
        # How far a sub-agent of this agent may go (config.settings.SUBAGENT_LEVELS).
        # The user's call, persisted like the toggles above; the weakest rung when they
        # have never chosen, because each rung hands something over rather than taking
        # it away.
        self.subagent_level = load_subagent_level()

    def apply_setting(self, name: str, value: Any) -> Any:
        """Change one of this agent's settings (see :mod:`core.settings`); returns the
        value now in force. Raises ``ValueError`` for a value the setting refuses."""
        return apply_setting(self, name, value)

    @staticmethod
    def normalize_workspace_path(path: str | None) -> str:
        return normalize_workspace_path(path)

    def normalize_tool_arguments(self, tool_name: str, arguments: dict) -> dict:
        path_arg_names = path_args(tool_name, self.tool_caps)
        return normalize_tool_arguments(
            tool_name,
            arguments,
            path_args_by_tool={tool_name: path_arg_names} if path_arg_names else {},
            normalize_tool_path_argument_fn=normalize_tool_path_argument,
        )

    def rewrite_tool_for_context(self, tool_name: str, arguments: dict) -> tuple[str, dict]:
        return rewrite_tool_for_context(
            tool_name,
            arguments,
            tool_owner=self.tool_owner,
            is_code_filepath=self.is_code_filepath,
            normalize_workspace_path_fn=self.normalize_workspace_path,
        )

    @staticmethod
    def new_execution_context() -> ExecutionContext:
        context = ExecutionContext()
        validate_execution_context(context)
        return context

    @staticmethod
    def is_code_filepath(path: str) -> bool:
        return os.path.splitext(path)[1].lower() in SOURCE_FILE_EXTENSIONS

    def approval_scope(self, tool_name: str, arguments: dict) -> str:
        """The normalised approval scope of this call ("" if none): the key of "always" grants and of refusals."""
        try:
            return self.approvals._approval_scope(
                tool_name, self.tool_owner.get(tool_name, "unknown"), arguments
            )
        except Exception:
            logger.debug("approval_scope: a best-effort step failed and was skipped", exc_info=True)
            return ""

    def record_denied_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        execution_context: ExecutionContext | None,
        note: str = "",
    ) -> None:
        """Record a refusal in the open set (``denied_tool_calls``) and the scope history.

        The open set clears when the action later succeeds; ``denial_history`` does not, since
        the escalation ladder counts it.
        """
        if execution_context is None:
            return
        fallback_tools = list(self.approvals.fallback_suggestions(tool_name))
        scope = self.approval_scope(tool_name, arguments)
        kind = denial_kind(note)
        execution_context.denials.denied_tool_calls.append({
            "tool": tool_name,
            "path": self.normalize_workspace_path(arguments.get("path") or arguments.get("filepath")),
            "fallback_tools": fallback_tools,
            "scope": scope,
            "kind": kind,
            "reason": note,
        })
        execution_context.denials.denial_history.append({
            "tool": tool_name,
            "scope": scope,
            "kind": kind,
            "reason": note,
        })

    async def auto_validate_written_file(self, path: str, execution_context: ExecutionContext | None) -> str:
        return await auto_validate_written_file(
            path=path,
            execution_context=execution_context,
            tool_owner=self.tool_owner,
            run_tool=lambda tool, args, context: self.run_tool(tool, args, execution_context=context),
            is_code_filepath=self.is_code_filepath,
            absolute_workspace_path_fn=absolute_workspace_path,
        )

    # Canonical argument key names that carry a target file path.

    def is_write_tool(self, tool_name: str) -> bool:
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
            raw = target_path_arg(tool_name, arguments, self.tool_caps)
            normalized = self.normalize_workspace_path(raw) if raw else ""
            return [normalized] if normalized else []


    def todo_file(self) -> str:
        """This agent's own todo_list.md, or '' when it has no planning tool.

        From ``self.session_id``; an agent without a session resolves through the pointer.
        """
        if not names_with_cap(TASK_PLANNING, self.tool_caps):
            return ""
        return os.path.join(
            session_state_dir(STATE_DIR, self.session_id), TODO_FILENAME
        )

    async def build_system_content(self, active_mode: str) -> str:
        return self.build_system_content_now(active_mode)

    def build_system_content_now(self, active_mode: str) -> str:
        """The full system prompt for *active_mode*, assembled synchronously.

        For a caller off the agent's own loop (the context bar measures the prompt from
        the WebSocket loop). Nothing here awaits: the assembly only reads local files.
        """
        # Both indexes under one gate: the memory server owns the two scopes together,
        # so either both are injectable or neither is.
        has_memory = "memory_search" in self.tool_owner
        memory_file = os.path.join(STATE_DIR, "memory", "MEMORY.md") if has_memory else ""
        global_memory_file = (
            os.path.join(GLOBAL_STATE_DIR, "memory", "MEMORY.md") if has_memory else ""
        )
        todo_file = self.todo_file()

        return build_system_content(
            active_mode=active_mode,
            tool_owner=self.tool_owner,
            sensitive_tools=self.approvals.sensitive_tools,
            memory_context_file=memory_file,
            global_memory_context_file=global_memory_file,
            todo_file=todo_file,
            plan_todos=self.plan_todos,
            thinking_depth=self.thinking_depth,
            session_id=self.session_id,
            delegation_available=bool(names_with_cap(DELEGATE, self.tool_caps)),
            subagent_level=self.subagent_level,
            workspace_root=self.workspace_root,
            # Read from the live schemas rather than restated anywhere: the server that
            # declares a tool owns what it does, and a catalog copied by hand is a
            # catalog that goes stale without anyone noticing.
            tool_descriptions={
                fn.get("name", ""): fn.get("description", "")
                for fn in (t.get("function", {}) for t in self.tools)
                if fn.get("name")
            },
            skills_index=self.model_invocable_skills(),
            model_name=self.model,
        )

    def model_invocable_skills(self) -> list[tuple[str, str]]:
        """``(name, description)`` of every skill the model may load, sorted (stable prefix).

        Excludes skills switched off and those declaring ``disable-model-invocation``.
        """
        return [
            (name, str(meta.get("description") or "").strip())
            for name, meta in sorted(self.skills.items())
            if self.skill_enabled(name) and meta.get("model_invocable", True)
        ]

    # ── The person behind the front-end (core.ports.HumanIO) ─────────────────

    def request_tool_approval(self, tool_name: str, arguments: dict,
                              max_attempts: int = 3) -> tuple[bool, str]:
        return self.human.approve_tool(self, tool_name, arguments, max_attempts)

    def request_path_approval(self, paths: list[str], tool_name: str,
                              arguments: dict | None = None) -> tuple[bool, bool]:
        """Approve out-of-workspace access to *paths*. Returns (approved, always)."""
        return self.human.approve_paths(self, paths, tool_name, arguments)

    def request_user_question(self, questions: list, origin: dict | None = None,
                              timeout_secs: float | None = None) -> dict:
        """Ask the user *questions* (``{header, question, multiSelect, options}`` specs).

        *origin* names an asker other than the turn on screen; *timeout_secs* None waits
        indefinitely. Returns ``{"answers": [...]}``, flagged ``timed_out`` when the wait ran out.
        """
        return self.human.ask(self, questions, origin, timeout_secs)

    def denied_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        note: str = "",
        execution_context: ExecutionContext | None = None,
    ) -> str:
        """The tool result a refused approval puts back in front of the model.

        Not "ask again": a refusal is an instruction, and the hint states the three
        readings it can carry so the model picks one instead of retrying. The stage
        (see guardrails.workflow) narrows that choice as refusals accumulate.
        """

        scope = self.approval_scope(tool_name, arguments)
        context = execution_context if execution_context is not None else ExecutionContext()
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

        return json_error_payload(
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

    async def connect_servers(
        self,
        registry: dict[str, str],
        on_error: Callable[[str, BaseException], None] | None = None,
    ) -> None:
        """Connect a whole server registry at once — the startup path.

        Prefer this to a loop over :meth:`connect_server`: the handshakes overlap,
        which is the difference between a registry costing the sum of its servers'
        start-up and costing the slowest one.
        """
        await connect_servers_runtime(agent=self, registry=registry, on_error=on_error)

    async def read_resource(self, uri: str) -> str:
        """The text of an MCP resource by URI, from its owning session; "" when unknown or unreadable."""
        info = self.resources.get(uri)
        if not info:
            return ""
        session = self.sessions.get(info.get("session", ""))
        if session is None:
            return ""
        try:
            result = await session.read_resource(AnyUrl(uri))
        except Exception:
            logger.debug("read_resource: a best-effort step failed and was skipped", exc_info=True)
            return ""
        parts = [
            c.text for c in getattr(result, "contents", [])
            if getattr(c, "text", None) is not None
        ]
        return "\n".join(parts)

    def seed_classification_from_caps(self) -> None:
        """Seed the approval manager from the connected tools' declared capabilities.

        Call once all servers are connected. The sets are updated in place: other objects hold
        references to them.
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
        off = self.toggles.disabled["server"]
        if not off:
            return self.tools
        return [
            t for t in self.tools
            if self.tool_owner.get(t.get("function", {}).get("name", "")) not in off
        ]

    def advertised_tools_for_mode(self, active_mode: str | None = None) -> list[dict]:
        """:meth:`advertised_tools` narrowed by the mode's read-only filter, as the loop applies it."""
        mode = self.mode if active_mode is None else active_mode
        tools = self.advertised_tools()
        if mode in READONLY_MODES:
            tools = tools_for_readonly_mode(tools, self.tool_caps, mode=mode)
        return tools

    def skill_enabled(self, name: str) -> bool:
        return self.toggles.enabled("skill", name)

    def set_enabled(self, kind: str, name: str, enabled: bool) -> None:
        """Switch a server, skill or nudge on or off, and remember it."""
        self.toggles.set_enabled(kind, name, enabled)

    def toggles_state(self) -> dict:
        """Every server (connected or not) and skill with its enabled flag and description, for the panel."""
        server_descriptions = all_server_descriptions()
        servers = [
            {
                "name": name,
                "description": server_descriptions.get(name, ""),
                "enabled": self.toggles.enabled("server", name),
            }
            for name in all_servers()
        ]
        skills = [
            {
                "name": name,
                # A skill its own file keeps for the user says so on the row: switched
                # on, it is still reachable only by /<name>, and a panel that showed it
                # as plainly "active" would read as "the model may use this".
                "description": (
                    meta.get("description", "")
                    if meta.get("model_invocable", True)
                    else f"{meta.get('description', '')} (/{name} only)".strip()
                ),
                "enabled": self.toggles.enabled("skill", name),
            }
            for name, meta in sorted(self.skills.items())
        ]
        # Application nudges from registered extension packs. Locked application
        # policies are intentionally NOT listed — they cannot be toggled off.
        nudges = [
            {
                "name": name,
                "description": "application nudge",
                "enabled": self.toggles.enabled("nudge", name),
            }
            for name in self.extensions.nudge_names()
        ]
        return {"servers": servers, "skills": skills, "nudges": nudges}

    def report_capability_consistency(self) -> None:
        """Log the connected tools that declare no capabilities (pure, or a forgotten ``tool_caps``)."""
        unannotated = unannotated_live_tools(self.tool_caps)
        if unannotated:
            logger.info(
                "Tool-capability registry: %d connected tool(s) declared no "
                "capabilities (pure tool, or missing tool_caps?): %s",
                len(unannotated), ", ".join(unannotated),
            )

    async def run_tool(
        self,
        tool_name: str,
        arguments: dict,
        execution_context: ExecutionContext | None = None,
        run_auto_validation: bool = True,
        call_id: str = "",
        record_observations: bool = True,
    ) -> str:
        return await execute_tool_call(
            agent=self,
            tool_name=tool_name,
            arguments=arguments,
            execution_context=execution_context,
            run_auto_validation=run_auto_validation,
            call_id=call_id,
            record_observations=record_observations,
        )

    async def compact_history(self, history: list[dict]) -> str:
        """Summarize *history* into one message with the configured backend; "" when empty or failed."""
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

        try:
            msg = get_backend().chat(
                self.model, messages, [], False, False, {"temperature": 0.2},
                token_callback=_discard_token,
            )
        except Exception:
            logger.debug("compact_history: a best-effort step failed and was skipped", exc_info=True)
            return ""
        return (msg or {}).get("content") or "" if isinstance(msg, dict) else ""

    def compact_messages(self, middle: list[dict]) -> list[dict]:
        """Summarize *middle* into one message; synchronous, for the loop's in-query compaction.

        The request is force-fit to the window first; on failure *middle* comes back unchanged.
        """
        if not middle:
            return middle

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
            force_fit_to_window(prompt, int(window * 0.75), tok)
        try:
            msg = backend.chat(
                self.model, prompt, [], False, False, {"temperature": 0.2},
                token_callback=_discard_token,
            )
        except Exception:
            logger.debug("compact_messages: a best-effort step failed and was skipped", exc_info=True)
            return middle
        summary = (msg or {}).get("content", "") if isinstance(msg, dict) else ""
        if not summary:
            return middle
        # compacted_exchanges, not len(middle) // 2: on a second pass *middle* opens on
        # the previous summary, which stands for far more than the one message it is.
        return [compaction_summary_message(compacted_exchanges(middle), summary)]

    def load_skills(self, skills_dir: str, *, merge: bool = False) -> None:
        """Load the skills of *skills_dir*. With *merge*, add them to those already
        loaded, a same-named one overriding (the workspace's over the bundled ones)."""
        if not merge:
            self.skills = {}
        for name, meta in read_skills_dir(skills_dir).items():
            if merge and name in self.skills:
                logger.info("User skill '%s' overrides the bundled skill of the same name", name)
            self.skills[name] = meta

    async def run(
        self,
        query: str,
        max_steps: int = 0,  # 0 = no ceiling; the run ends when the work does
        history: list[dict] | None = None,
        mode: str | None = None,
        thinking: bool = False,
        streaming: bool = True,
        token_callback: Any = None,
        think_token_callback: Any = None,
        think_start_callback: Any = None,
        think_end_callback: Any = None,
        event_callback: Any = None,
        resume: dict | None = None,
    ) -> str:
        # With an event_callback, emit() in the engine and the executor (and their
        # gathered tasks) goes to it for this run; without one (the CLI) it prints.
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
                resume=resume,
            )
        finally:
            if token is not None:
                reset_event_sink(token)

    async def cleanup(self) -> None:
        await self.exit_stack.aclose()

    # ── Session state, as the session store keeps it ───────────────────────────

    def export_state(self) -> dict[str, Any]:
        return {"carry_context": self.carry.to_json()}

    def load_state(self, state: dict[str, Any]) -> None:
        self.carry = CarryContext.from_json(state.get("carry_context", {}))


if TYPE_CHECKING:
    from .core.ports import AgentPort

    # mypy holds MimirAgent to the port every layer is handed.
    _agent_is_a_port: type[AgentPort] = MimirAgent


__all__ = ["MimirAgent", "SERVERS", "_BASE"]

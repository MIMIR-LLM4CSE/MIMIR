"""Built-in call-time policy gates.

Each gate is a function ``check_*(agent, tool_name, [arguments,] execution_context)
-> str | None`` returning a violation payload (``json_error_payload``) to BLOCK, or
``None`` to allow. :mod:`.engine` runs them in a fixed order and enriches what they
return; the gates know nothing of the orchestration.
"""

from __future__ import annotations

from ...context.context_view import ContextView
from ...context.execution_context import ExecutionContext
import logging
import os
from typing import Any

from ....shared.shell_paths import (
    allocates_cluster,
    cd_destination,
    normalize_path_arg,
    parse_segments,
    segment_path_operands,
)
from ....shared.trusted_read_roots import trusted_read_roots
from ...config import paths
from ...config.limits import SKILL_PULL_MAX
from ...config.servers import SKILL_LOAD_TOOL
from ...tools.bash_classify import segment_program, shell_segments
from ...tools.capabilities import (
    CLUSTER_SUBMIT,
    EDIT,
    PLAN_BLOCKED,
    PLAN_READONLY,
    READ,
    arg_role,
    has_cap,
    label_for,
    shell_command_args,
)
from ...context.paths import absolute_workspace_path, scratch_roots
from ...core.ports import PolicyAgent
from ....shared import exec_seals
from ...protocol.tool_result import json_error_payload
from ...tools.display import shorten_display_args, tool_arg_preview, tool_status_message

logger = logging.getLogger(__name__)

# ── unattended agents ─────────────────────────────────────────────────────────

def describe_action(agent: PolicyAgent, tool_name: str, arguments: dict) -> str:
    """The action as the user would read it on a tool row — never the tool's name."""
    short_args = shorten_display_args(tool_name, arguments or {}, agent.tool_caps)
    label = label_for(tool_name, short_args, agent.tool_caps)
    if label:
        return label
    preview = tool_arg_preview(tool_name, arguments or {})
    status = tool_status_message(tool_name, arguments or {})
    return f"{status}: {preview}" if preview else status


def mode_denied_payload(agent: PolicyAgent, tool_name: str, arguments: dict, needs: str) -> str:
    """The result an unattended agent gets for an action its mode does not cover.

    Deterministic, not a reading to choose among: no one refused, so there is no
    "another way" to look for. The agent skips the step or stops, and says so.
    """
    agent.approvals.block_for_mode(describe_action(agent, tool_name, arguments), needs)
    return json_error_payload(
        f"Not run: the approval mode is '{agent.approvals.approval_mode}', this action "
        f"needs '{needs}', and a sub-agent cannot ask the user.",
        hint=("Do not retry it and do not reach the same result another way. If the rest "
              "of the task can go on without this step, skip it and continue; otherwise "
              "stop now. Either way, name the skipped action in your answer."),
        tool=tool_name,
        denial_kind="mode_denied",
        approval_mode=agent.approvals.approval_mode,
        needs_mode=needs,
    )


# ── out-of-workspace access ───────────────────────────────────────────────────

def _is_under(root: str, full: str) -> bool:
    return full == root or full.startswith(root + os.sep)


def _trusted_read_roots() -> list[str]:
    """Roots read without prompting: the servers' trusted roots plus ``paths.STATE_DIR``.

    The client process has no ``MIMIR_STATE_DIR``, so the state dir is added here.
    """
    return [os.path.realpath(os.path.abspath(r))
            for r in (*trusted_read_roots(), paths.STATE_DIR)]


def _shell_path_targets(agent: PolicyAgent, tool_name: str, arguments: dict, base: str) -> list[str]:
    """Every filesystem path a shell call names, as abspaths: operands, redirections, ``cd``.

    Segmented by the server's own ``parse_segments`` and ``segment_path_operands``, so what
    the user is asked about is exactly what the server would confine. Scoped to tools with
    a ``command_prefix`` scope.
    """
    if shell_command_args(tool_name, agent.tool_caps) is None:
        return []

    out: list[str] = []
    for val in arguments.values():
        if not isinstance(val, str) or not val.strip():
            continue
        try:
            # A bare $VAR blocks classification but not path extraction: a command
            # mixing one with a real out-of-workspace path must still reach the user.
            # parse_segments keeps such a command, so nothing here has to allow for it.
            segments = parse_segments(val)
        except Exception:
            # A ShellParseError, or anything unforeseen: a command the shared parser
            # will not read is one the server refuses before any shell opens, so there
            # is no access here to ask the user about.
            continue
        cursor = base
        for segment in segments:
            argv = segment.argv
            for target in segment.redirect_targets:
                path = normalize_path_arg(target, cursor)
                if path is not None and path not in out:
                    out.append(path)
            if argv and argv[0] == "cd":
                cursor = cd_destination(argv, cursor, base)
                if cursor not in out:
                    out.append(cursor)
                continue
            for path in segment_path_operands(argv, cursor):
                if path not in out:
                    out.append(path)
    return out


def _call_base(agent: PolicyAgent, tool_name: str, arguments: dict) -> tuple[str, str]:
    """``(root, base)``: the agent's workspace root, and where this call's relative
    paths resolve — a declared ``cwd`` arg-role where the tool has one, else the root.
    """
    # The agent's own root: a sub-agent working in a copy of the repository is inside
    # ITS workspace there, and outside the caller's — which is the whole point of the
    # copy. Falls back to the module constant for anything agent-like without one.
    root = os.path.realpath(os.path.abspath(
        getattr(agent, "workspace_root", None) or paths.WORKSPACE_ROOT))
    # The bash server has no cwd role — every call starts at the workspace root — so
    # this normally leaves the base at the root.
    base = root
    for a in arg_role(tool_name, "cwd", agent.tool_caps):
        v = arguments.get(a)
        if isinstance(v, str) and v.strip():
            base = os.path.realpath(absolute_workspace_path(v, cwd=root))
    return root, base


def out_of_workspace_targets(agent: PolicyAgent, tool_name: str, arguments: dict) -> list[str]:
    """Absolute target paths of this call outside the workspace and not yet approved.

    File targets, a declared ``cwd``, and the paths inside a shell command; trusted read
    roots excluded.
    """
    root, call_base = _call_base(agent, tool_name, arguments)

    raw: list[str] = []
    try:
        raw.extend(agent.get_tool_file_targets(tool_name, arguments) or [])
    except Exception:
        logger.debug("out_of_workspace_targets: a best-effort step failed and was skipped", exc_info=True)
    # A declared cwd is itself a place the call goes.
    for a in arg_role(tool_name, "cwd", agent.tool_caps):
        v = arguments.get(a)
        if isinstance(v, str) and v.strip():
            raw.append(v)
    raw.extend(_shell_path_targets(agent, tool_name, arguments, call_base))

    is_read = has_cap(tool_name, READ, agent.tool_caps) and not (
        agent.is_write_tool(tool_name)
        or has_cap(tool_name, EDIT, agent.tool_caps)
        or has_cap(tool_name, PLAN_BLOCKED, agent.tool_caps))
    trusted = _trusted_read_roots()

    scratch = scratch_roots()

    out: list[str] = []
    for p in raw:
        full = os.path.realpath(absolute_workspace_path(p, cwd=root))
        if _is_under(root, full):
            continue
        # The scratchpad is outside the workspace by design, and prompting for it
        # would defeat its purpose — it exists precisely so throwaway work has a
        # home that costs no user decision.
        if any(_is_under(s, full) for s in scratch):
            continue
        if is_read and any(_is_under(t, full) for t in trusted):
            continue
        if agent.approvals.paths.covers(full):
            continue
        if full not in out:
            out.append(full)
    # One command routinely names a directory *and* paths inside it (``cd /data &&
    # python run.py``). Approving the directory grants it to the server as a root,
    # so its children are allowed the moment the parent is — prompting for each in
    # turn asks the same decision several times over.
    return [t for t in out if not any(_is_under(other, t) for other in out if other != t)]


def _name_paths(paths: list[str], limit: int = 4) -> str:
    """The refused paths, quoted — capped so a wide command doesn't flood the error."""
    shown = ", ".join(f"'{p}'" for p in paths[:limit])
    rest = len(paths) - limit
    return f"{shown} and {rest} more" if rest > 0 else shown


def check_out_of_workspace_access(
    agent: PolicyAgent, tool_name: str, arguments: dict, execution_context: ContextView | None,
    targets: list[str] | None = None,
) -> str | None:
    """Ask the user once about every outside path of this call; deny when no one can answer.

    *targets* is the list already computed by :func:`out_of_workspace_targets`. Returns a
    violation payload or ``None``.
    """
    if targets is None:
        targets = out_of_workspace_targets(agent, tool_name, arguments)
    if not targets:
        return None
    # ``auto_all``: grant without asking. The servers still validate and confine every
    # accepted call, so this suppresses the question, not the sandbox.
    if agent.approvals.auto_paths():
        for abspath in targets:
            agent.approvals.paths.grant(abspath, always=False)
        return None
    if agent.approvals.unattended:
        return mode_denied_payload(agent, tool_name, arguments, "auto_all")
    prompt = getattr(agent, "request_path_approval", None)
    # The call's own arguments travel with the paths so the prompt can show the
    # usual tool description ("Running: …") instead of bare paths.
    approved, always = prompt(targets, tool_name, arguments) if prompt else (False, False)
    if not approved:
        return json_error_payload(
            f"Access outside the workspace was not approved: {_name_paths(targets)}.",
            hint=("These paths are outside the workspace root. Ask the user to approve "
                  "them (allow / always), or operate within the workspace."),
            tool=tool_name,
        )
    for abspath in targets:
        agent.approvals.paths.grant(abspath, always=always)
    return None


# ── cluster submit ────────────────────────────────────────────────────────────

def check_cluster_submit(
    agent: PolicyAgent, tool_name: str, execution_context: ContextView | None,
    arguments: dict[str, Any] | None = None,
) -> str | None:
    """Hold a ``CLUSTER_SUBMIT`` call until something has been validated locally.

    Not one-shot: it holds until a file passes a check. A session that wrote nothing is
    exempt. Returns a violation payload or ``None`` (POLICY.md, "Cluster-submission guard").
    """
    if execution_context is None:
        return None
    # The dual-use shell submits too when its command allocates (``srun`` outside an
    # allocation): same cost, same precondition. Recognised by capability and by what
    # the command does, never by the tool's name.
    shell_submit = (has_cap(tool_name, PLAN_READONLY, agent.tool_caps)
                    and allocates_cluster(str((arguments or {}).get("command", ""))))
    if not (has_cap(tool_name, CLUSTER_SUBMIT, agent.tool_caps) or shell_submit):
        return None
    if execution_context.checks.validated_files:
        return None
    if not execution_context.edits.dirty_written_files:
        return None
    return json_error_payload(
        f"Cluster submission '{tool_name}' held: nothing validated locally yet this session.",
        hint=(
            "Cluster submissions consume real allocation hours and cannot be taken back. "
            "Validate the script or command locally first (a syntax/import check or a quick "
            "local smoke run) so a trivial error doesn't waste a job. This hold lifts as soon "
            "as one local check passes — retrying without one will not clear it."
        ),
        tool=tool_name,
    )


# ── sealed-execution guard ────────────────────────────────────────────────────
# A server seals the files that must run only through its own tools (shared/exec_seals);
# running one directly is refused in the seal's own words.

def _executed_programs(command: str, base: str) -> set[str]:
    """Real paths of the programs a shell command (or a bare path) would execute.

    Relative programs resolve against *base*; a command the server's parser refuses yields
    the empty set.
    """
    if not isinstance(command, str) or not command.strip():
        return set()
    segments = shell_segments(command, allow_expansion=True) or []
    progs: set[str] = set()
    for seg in segments:
        prog = segment_program(list(seg.argv))
        if prog:
            progs.add(os.path.realpath(os.path.join(base, os.path.expanduser(prog))))
    return progs

def check_sealed_exec(
    agent: PolicyAgent, tool_name: str, arguments: dict[str, Any],
    execution_context: ContextView | None,
) -> str | None:
    """Refuse running a file a server sealed for this conversation (fail-open).

    Only for tools with a ``command_prefix`` scope, and only a program in command position;
    reading a sealed file never matches (POLICY.md, "Sealed-execution guard").
    """
    if shell_command_args(tool_name, agent.tool_caps) is None:
        return None
    try:
        seals = exec_seals.sealed(getattr(agent, "session_id", None))
        if not seals:
            return None
        _root, base = _call_base(agent, tool_name, arguments)
        executed: set[str] = set()
        for val in arguments.values():
            if isinstance(val, str) and val.strip():
                executed |= _executed_programs(val, base)
        hit = next((seals[p] for p in sorted(executed) if p in seals), None)
        if hit is None:
            return None
    except Exception:
        logger.debug("check_sealed_exec: a best-effort step failed and was skipped", exc_info=True)
        return None
    return json_error_payload(
        f"{str(hit.get('refusal') or 'This file is sealed: it runs only through its server.')} "
        f"('{tool_name}')",
        hint=str(hit.get("hint") or ""),
        tool=tool_name,
    )


# ── loading a skill on demand ─────────────────────────────────────────────────

def check_skill_pull(
    agent: PolicyAgent, tool_name: str, arguments: dict[str, Any],
    execution_context: ExecutionContext | None,
) -> str | None:
    """Refuse a skill pull the toggles, the skill itself, or the budget rules out.

    Reserves the slot (``"pending"``) on the way, so two pulls in one step cannot both pass
    the cap; the dispatcher promotes or releases it.
    """
    if tool_name != SKILL_LOAD_TOOL:
        return None
    skill = str((arguments or {}).get("name") or "").strip()
    if not skill:
        return None   # the server answers for a malformed name
    # A call made outside a query has nothing loaded and nothing to reserve against.
    loaded: dict = execution_context.bookkeeping.skills_loaded if execution_context is not None else {}

    if skill in loaded:
        return json_error_payload(
            f"The skill '{skill}' is already loaded in this conversation.",
            hint="Its instructions are in an earlier tool result — re-read that result "
                 "rather than loading it again.",
            tool=tool_name,
        )

    if len(loaded) >= SKILL_PULL_MAX:
        return json_error_payload(
            f"{len(loaded)} skills are already loaded "
            f"({', '.join(sorted(loaded))}).",
            hint="That is the limit for one task. Finish the work with the methods you "
                 "have; a fourth playbook would not be applied either.",
            tool=tool_name,
        )

    available = dict(getattr(agent, "skills", None) or {})
    meta = available.get(skill)
    offered = []
    fn = getattr(agent, "model_invocable_skills", None)
    if callable(fn):
        offered = [name for name, _desc in fn()]

    # A skill the operator soft-hid gets the SAME answer as one that does not exist,
    # deliberately. It is absent from the index the model was given, so naming it as
    # "switched off" tells the model about a capability it cannot have and invites it to
    # stop and ask the user to switch it back on — a worse outcome than not knowing.
    if meta is None or not agent.skill_enabled(skill):
        return json_error_payload(
            f"No skill named '{skill}'.",
            hint=(f"Available: {', '.join(offered)}." if offered
                  else "No skills are available to load."),
            tool=tool_name,
        )

    if not meta.get("model_invocable", True):
        return json_error_payload(
            f"The skill '{skill}' is user-invoked only.",
            hint=f"The user can run /{skill}. Carry on with your own method — do not "
                 "ask them to run it for you.",
            tool=tool_name,
        )

    loaded[skill] = "pending"
    return None

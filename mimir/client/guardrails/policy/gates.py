"""Built-in call-time policy gates.

Each gate is a small function ``_check_*(agent, tool_name, [arguments,]
execution_context) -> str | None`` returning a raw violation payload (built via
``agent._json_error_payload``) to BLOCK, or ``None`` to allow. The orchestrator
in :mod:`.engine` runs them in a fixed order and enriches the returned payload;
the gates themselves stay free of orchestration concerns.

These moved out of ``engine.py`` (which had grown to several jobs) so the gates
live together and a new one lands next to its siblings rather than in a
one-policy module. Behaviour is unchanged for the three original gates.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Any

from ....servers._shared.shell_paths import COMMAND_WRAPPERS

from ...context.capabilities import (
    CLUSTER_SUBMIT, EDIT, PLAN_BLOCKED, READ, TASK_PLANNING,
    arg_role, has_cap, scope_spec,
)


# ── out-of-workspace access ───────────────────────────────────────────────────

def _is_under(root: str, full: str) -> bool:
    return full == root or full.startswith(root + os.sep)


def _trusted_read_roots() -> list[str]:
    """Client mirror of the servers' silent read roots (item 1): proxy/HPC caches
    and the central state dir. Reads of these never prompt. Shares its source of
    truth with the servers via ``servers._shared.trusted_read_roots`` so the two
    lists can't drift.

    ``STATE_DIR`` is appended explicitly because the shared helper resolves it from
    ``MIMIR_STATE_DIR``, and that variable is only ever placed in the *server*
    subprocesses' env (server_manager) — the client process itself does not carry
    it. Without this the agent could not read back its own plans/sessions without a
    prompt, while the servers, which do see the variable, would have allowed it.
    """
    from ....servers._shared.trusted_read_roots import trusted_read_roots
    from ...config.constants import STATE_DIR
    return [os.path.realpath(os.path.abspath(r))
            for r in (*trusted_read_roots(), STATE_DIR)]


def _shell_command_args(agent: Any, tool_name: str) -> tuple[str, ...] | None:
    """Command-arg names if *tool_name* takes a raw shell command, else None.

    Registry-driven: a shell tool declares a ``command_prefix`` scope kind. The twin of
    ``observations._carries_shell_command`` — the guards below read tool arguments *as
    shell*, which is a statement about the tool's interface, not about whether it
    executes.
    """
    spec = scope_spec(tool_name, getattr(agent, "tool_caps", None))
    if not spec or spec.get("kind") != "command_prefix":
        return None
    return tuple(spec.get("args") or ("command",))


def _shell_path_targets(agent: Any, tool_name: str, arguments: dict, base: str) -> list[str]:
    """Every filesystem path a shell call's arguments name, as abspaths.

    A shell command carries its paths inside a string, so the ordinary file-target
    extractor cannot see them: without this, ``cat /etc/passwd`` was refused by the
    server with no prompt ever shown, which left the user unable to grant an access
    they might well have wanted. Walking the segments here routes every such path
    through the ordinary out-of-workspace prompt — approve it and the grant reaches
    the server's sandbox guard through the shared sidecar, so the command runs.

    A ``cd`` is included for a subtler reason: it is side-effect-free in itself, but
    stepping outside the workspace changes what every later relative path in the
    chain resolves to. So the destination is threaded through the walk (chained
    ``cd``s accumulate) and surfaced as a target of its own. A ``cd`` that stays
    inside the workspace yields nothing — the approval for such a chain is decided
    by the command that follows it, as for any other call.

    A redirection target counts as one of those paths: ``./solver.out > run.log``
    creates a file exactly as a ``-o`` flag would, and the server confines it the
    same way, so the user must be asked about it on the same terms.

    Operand extraction is ``shell_paths.segment_path_operands``, the routine the
    server's guard walks to decide what to *confine*, so what the user is asked
    about is exactly what would otherwise be refused. Scoped to tools that declare a
    ``command_prefix`` scope — the property this actually needs, since it reads the
    arguments *as shell*. CODE_EXEC would be the wrong test: it also marks tools that
    execute through structured arguments (``proxy_exec``, ``ft_run``), whose parameters
    are not a command line. Driven off the shared segmenter, so no tool name or shell
    keyword is spelled out here. Fail-open on an unparseable command: the server still
    validates and confines every accepted call independently.
    """
    if _shell_command_args(agent, tool_name) is None:
        return []
    from ....servers._shared.shell_paths import (
        cd_destination, normalize_path_arg, segment_path_operands,
    )
    from .bash_classify import shell_segments

    out: list[str] = []
    for val in arguments.values():
        if not isinstance(val, str) or not val.strip():
            continue
        try:
            # allow_expansion: a bare $VAR blocks classification but not path
            # extraction, and a command mixing one with a real out-of-workspace
            # path must still reach the user (see shell_segments).
            segments = shell_segments(val, allow_expansion=True)
        except Exception:
            continue
        cursor = base
        for segment in segments or []:
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


def _out_of_workspace_targets(agent: Any, tool_name: str, arguments: dict) -> list[str]:
    """Absolute target paths of this call outside the workspace and not yet approved.

    Sources: the existing file/edit-target extractor (covers file & batch args), any
    declared ``cwd`` arg-role, and every path named inside a shell command — file
    operands and ``cd`` destinations alike (see :func:`_shell_path_targets`), which
    the file-target extractor cannot see because they live inside a string. Reads of
    the trusted roots (item 1) and already-approved paths are excluded.
    """
    from ...config.constants import WORKSPACE_ROOT
    from ...tool_execution.validation import absolute_workspace_path
    root = os.path.realpath(os.path.abspath(WORKSPACE_ROOT))

    raw: list[str] = []
    try:
        raw.extend(agent.get_tool_file_targets(tool_name, arguments) or [])
    except Exception:
        pass
    # A declared cwd arg-role, where a tool still has one, is also the base its
    # relative paths resolve against. The bash server has none — every call starts
    # at the workspace root — so this normally leaves the base at the root.
    call_base = root
    for a in arg_role(tool_name, "cwd", getattr(agent, "tool_caps", None)):
        v = arguments.get(a)
        if isinstance(v, str) and v.strip():
            raw.append(v)
            call_base = os.path.realpath(absolute_workspace_path(v, cwd=root))
    raw.extend(_shell_path_targets(agent, tool_name, arguments, call_base))

    is_read = has_cap(tool_name, READ, agent.tool_caps) and not (
        agent._is_write_tool(tool_name)
        or has_cap(tool_name, EDIT, agent.tool_caps)
        or has_cap(tool_name, PLAN_BLOCKED, agent.tool_caps))
    trusted = _trusted_read_roots()

    from ...tool_execution.validation import scratch_roots
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
        if agent.approvals.is_path_approved(full):
            continue
        if full not in out:
            out.append(full)
    # One command routinely names a directory *and* paths inside it (``cd /data &&
    # python run.py``). Approving the directory grants it to the server as a root,
    # so its children are allowed the moment the parent is — prompting for each in
    # turn asks the same decision several times over.
    return [t for t in out if not any(_is_under(other, t) for other in out if other != t)]


def _check_out_of_workspace_access(
    agent: Any, tool_name: str, arguments: dict, execution_context: dict[str, Any] | None,
    targets: list[str] | None = None,
) -> str | None:
    """Prompt the user before a tool touches a path outside the workspace.

    Reuses the existing approval UI (allow / deny / always-for-this-file, via the
    agent's ``_request_path_approval`` hook). Runs before the sensitive/preview gate
    so a previewable write cannot bypass it. Fail-closed: with no hook wired,
    out-of-workspace access is denied. Returns a violation payload or ``None``.

    *targets* lets the caller pass the list it already computed with
    :func:`_out_of_workspace_targets` — the engine needs to know whether this call
    prompts at all, and re-deriving it after the grants land would come back empty.
    """
    if targets is None:
        targets = _out_of_workspace_targets(agent, tool_name, arguments)
    if not targets:
        return None
    prompt = getattr(agent, "_request_path_approval", None)
    for abspath in targets:
        # The call's own arguments travel with the path so the prompt can show the
        # usual tool description ("Running: …") instead of a bare path.
        approved, always = prompt(abspath, tool_name, arguments) if prompt else (False, False)
        if not approved:
            return agent._json_error_payload(
                f"Access to '{abspath}' outside the workspace was not approved.",
                hint=("This path is outside the workspace root. Ask the user to approve "
                      "it (allow / always), or operate within the workspace."),
                tool=tool_name,
            )
        agent.approvals.grant_path(abspath, always=always)
    return None


# ── cluster submit ────────────────────────────────────────────────────────────

def _check_cluster_submit(
    agent: Any, tool_name: str, execution_context: dict[str, Any] | None
) -> str | None:
    """Hold an expensive cluster submission until something has been validated locally.

    Verification-tier guard (always on, never enforcement-tiered): a CLUSTER_SUBMIT
    tool (Slurm allocation / batch run) consumes real cluster hours, so the call
    requires some local-validation evidence (``validated_files`` — a file that passed a
    check) before it may run.

    **The hold is not one-shot.** It used to set a flag and let the very next retry
    through, which made it a reminder rather than a guard: against a model that simply
    calls again — the normal reaction to an error — it cost one round trip and
    constrained nothing, on the single most expensive action in the system. The
    condition is a fact about the session (was anything validated?), not a nagging
    budget, so it holds until that fact changes. The error names exactly what would
    clear it, so this is a precondition with a stated exit, not a wall.

    **A session that wrote nothing is exempt**, and that is what keeps the exit
    reachable. ``validated_files`` is credited only by a checker run against a file the
    model edited, so a run whose whole job is to launch something that already exists
    ("resubmit this job on 64 nodes") can never satisfy the condition however long it
    tries — the hold would be permanent, with the stated exit unreachable. Nothing was
    changed, so there is nothing that ought to have been checked; the user's own
    approval prompt, which these irreversible tools always raise, is the protection
    that applies there.

    Returns a structured violation payload, or ``None`` when the call is allowed.
    """
    if execution_context is None:
        return None
    if not has_cap(tool_name, CLUSTER_SUBMIT, agent.tool_caps):
        return None
    if execution_context.get("validated_files"):
        return None
    if not execution_context.get("dirty_written_files"):
        return None
    return agent._json_error_payload(
        f"Cluster submission '{tool_name}' held: nothing validated locally yet this session.",
        hint=(
            "Cluster submissions consume real allocation hours and cannot be taken back. "
            "Validate the script or command locally first (a syntax/import check or a quick "
            "local smoke run) so a trivial error doesn't waste a job. This hold lifts as soon "
            "as one local check passes — retrying without one will not clear it."
        ),
        tool=tool_name,
    )


# ── proxy direct-execution guard ──────────────────────────────────────────────
#
# In a proxy optimization session the model may only improve the proxy by editing its
# source and going through ``proxy_eval(op='run')``. A direct ``python proxy.py`` skips
# reference sealing, the numerical invariants and the ratchet, letting a hand-run be
# reported as a win. Scoped to shell-command tools; blocks only calls that *execute* the
# proxy (command position), never read-only inspection of its source. Locked and
# non-tiered — a correctness boundary, not guidance. Fail-open on internal error.

# Leading tokens that wrap another command; we look past them (and their flags /
# ``VAR=val`` assignments) to find the program actually executed. The shared set the
# bash validator unwraps with, plus ``time``, which is a shell keyword rather than a
# command and so never reaches that validator as a head.
_EXEC_WRAPPERS = COMMAND_WRAPPERS | {"time"}
# Interpreters that execute their first non-flag argument as the real program.
_EXEC_INTERPRETERS = frozenset({
    "python", "python3", "python2", "pypy", "pypy3", "sh", "bash", "zsh",
    "node", "ruby", "perl",
})
_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")


def _segment_program(argv: list[str]) -> str | None:
    """The program a single shell segment executes, or ``None``.

    Skips leading ``VAR=val`` env assignments and known wrappers (``env``/``time``/
    ``srun``…) with their flags, then returns the head — or, if the head is an
    interpreter, its first non-flag argument (``python proxy.py`` runs proxy.py).
    """
    i = 0
    while i < len(argv):
        tok = argv[i]
        if _ENV_ASSIGN_RE.match(tok) and not tok.startswith("/"):
            i += 1
            continue
        if os.path.basename(tok) in _EXEC_WRAPPERS:
            i += 1
            while i < len(argv) and argv[i].startswith("-"):
                i += 1
            continue
        break
    if i >= len(argv):
        return None
    head = argv[i]
    if os.path.basename(head) in _EXEC_INTERPRETERS:
        return next((a for a in argv[i + 1:] if not a.startswith("-")), None)
    return head


def _executed_programs(command: str) -> set[str]:
    """Programs a shell command (or a bare path) would execute.

    Returned as a set of abspaths *and* basenames so both ``python /abs/proxy.py``
    and ``python proxy.py`` match a proxy path. Mirrors the shlex segmentation of
    ``bash_classify.classify_bash_command`` (split on ``; && || |``, per-segment
    leading command). A bare path (e.g. running ``./proxy`` directly) parses to a
    single segment whose head is that path, so the same routine covers direct-path
    args. Fail-open: an unparseable payload yields the empty set.
    """
    if not isinstance(command, str) or not command.strip():
        return set()
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return set()

    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in (";", "&&", "||", "|"):
            if current:
                segments.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        segments.append(current)

    progs: set[str] = set()
    for argv in segments:
        prog = _segment_program(argv)
        if not prog:
            continue
        ap = os.path.abspath(prog)
        progs.add(ap)
        progs.add(os.path.basename(ap))
    return progs


def _proxy_exec_targets() -> set[str]:
    """Abspaths + basenames that count as "executing the proxy under optimization".

    Reads the proxy server's own store (single source of truth) for the currently
    active optimization session; returns an empty set when none is initialized so
    the guard abstains. Best-effort — any failure yields the empty set.
    """
    from ....servers.proxy._lib import store

    name = store._resolve_proxy_name("")
    if not name:
        return set()
    cfg = store._read_json(store._opt_config_file(name))
    if not isinstance(cfg, dict):
        return set()

    paths: set[str] = set()
    src = cfg.get("proxy_source_path")
    if isinstance(src, str) and src:
        paths.add(src)
    reg = store._read_json(store.registry_path())
    if isinstance(reg, dict):
        entry = reg.get(name)
        if isinstance(entry, dict):
            exe = entry.get("executable")
            if isinstance(exe, str) and exe:
                paths.add(exe)

    targets: set[str] = set()
    for p in paths:
        ap = os.path.abspath(p)
        targets.add(ap)
        targets.add(os.path.basename(ap))
    return targets


def _check_proxy_exec(
    agent: Any, tool_name: str, arguments: dict[str, Any],
    execution_context: dict[str, Any] | None,
) -> str | None:
    """Block direct execution of a proxy that is under optimization.

    Fast-path abstains for any tool that does not carry a shell command (no command
    line to read). Otherwise, if an optimization session is initialized, blocks the call
    only when one of its string arguments *executes* the proxy source/executable (in
    command position). Read-only inspection of the source (``cat``/``grep``) never
    matches. Fail-open.

    Scoped on the ``command_prefix`` scope rather than CODE_EXEC: the sanctioned route
    ``proxy_exec`` declares CODE_EXEC too, and its ``proxy_name`` argument is a bare
    name that :func:`_segment_program` would read as a program — matching the basenames
    in :func:`_proxy_exec_targets` and blocking the very tool this guard exists to steer
    the model towards.

    Note: this covers a bash ``python proxy.py`` / ``./proxy``. Executing the proxy
    by pasting its body inline into a fresh script is a narrower, documented residual
    gap; the reserved-metrics guard still prevents forging an accepted result there.
    """
    if _shell_command_args(agent, tool_name) is None:
        return None
    try:
        targets = _proxy_exec_targets()
        if not targets:
            return None
        executed: set[str] = set()
        for val in arguments.values():
            if isinstance(val, str) and val.strip():
                executed |= _executed_programs(val)
        if not (executed & targets):
            return None
    except Exception:
        return None
    return agent._json_error_payload(
        f"Direct execution of the proxy under optimization is blocked ('{tool_name}').",
        hint=(
            "A proxy optimization session is active. Run the proxy through "
            "proxy_eval(op='run') — executing it directly bypasses reference sealing, "
            "the numerical invariants and the ratchet, so a hand-run cannot be a valid "
            "result. To run it directly again, end the session with proxy_eval(op='reset')."
        ),
        tool=tool_name,
    )


# ── plan shape ────────────────────────────────────────────────────────────────
#
# PHASE 2 of the plan-mode prompt already states the rule — "exploring, surveying,
# examining, reviewing and identifying gaps ... are never steps or axes of the plan"
# — and nothing checked it. A plan whose first axis is "Audit the existing bindings"
# was observed in the wild: the audit then returned "nothing is missing", every axis
# after it was vacuous, and the model padded the run with cosmetic edits rather than
# re-deciding. The exploration has to be spent BEFORE the plan, or the plan is a
# guess about what the exploration will find.
#
# Only axis TITLES are read, never the prose: `PLAN_EXPLORE_BUDGET_SPENT` explicitly
# asks the model to state which assumptions it could not verify, and that sentence
# belongs in the body. Stating an open assumption is honest; making it an axis is not.

# Verbs that describe finding something out. A plan axis is a change to make, so a
# title leading with one of these names work that either is already done or belongs
# in the exploration phase.
# "map" and "list" are deliberately absent: both can head a real change ("map the old
# API onto the new one", "list the supported orders in the docs"), and a guard that
# refuses a legitimate axis costs a turn for nothing.
_PLAN_EXPLORATION_VERBS: frozenset[str] = frozenset({
    "analyse", "analyze", "assess", "audit", "catalog", "catalogue", "check",
    "compare", "determine", "diagnose", "evaluate", "examine", "explore",
    "identify", "inspect", "inventory", "investigate", "locate", "review",
    "study", "survey", "understand", "verify",
})
# The headings the tool's own docstring prescribes. They are structure, not axes:
# "## Validation" must never read as "Validate ...".
_PLAN_PRESCRIBED_HEADINGS: frozenset[str] = frozenset({
    "overview", "approach", "decisions", "decisions & risks", "decisions and risks",
    "risks", "validation", "context", "key files", "files",
})
_PLAN_AXIS_RE = re.compile(
    r"^\s*(?:#{3,6}\s*|[-*]\s+|\d+[.)]\s+)(?:\*\*)?\s*([^\n]{1,120}?)\s*(?:\*\*)?\s*$",
    re.MULTILINE,
)
_PLAN_APPROACH_RE = re.compile(r"^(#{1,3})[ \t]*approach\b.*$", re.IGNORECASE | re.MULTILINE)


def _leading_verb(title: str) -> str:
    """First alphabetic word of *title*, lowercased and de-suffixed to a bare verb."""
    words = re.findall(r"[A-Za-z]+", title)
    if not words:
        return ""
    word = words[0].lower()
    # "Auditing the bindings" / "Audits" are the same axis as "Audit the bindings".
    for suffix in ("ing", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            stem = word[: -len(suffix)]
            if stem in _PLAN_EXPLORATION_VERBS:
                return stem
            if suffix == "ing" and (stem + "e") in _PLAN_EXPLORATION_VERBS:
                return stem + "e"
    return word


def _approach_body(text: str) -> str:
    """The slice of *text* under its Approach heading, or "" when there is none.

    The section ends at the next heading of the SAME level or shallower — the axes
    themselves are deeper headings, and a fixed boundary cut the section at its own
    first axis, which is exactly the content this reads.
    """
    match = _PLAN_APPROACH_RE.search(text)
    if match is None:
        return ""
    level = len(match.group(1))
    rest = text[match.end():]
    nxt = re.compile(rf"^#{{1,{level}}}[ \t]+", re.MULTILINE).search(rest)
    return rest[: nxt.start()] if nxt else rest


def _exploration_axes(text: str) -> list[str]:
    """Axis titles under Approach that lead with an exploration verb."""
    offenders: list[str] = []
    for title in _PLAN_AXIS_RE.findall(_approach_body(text)):
        cleaned = title.strip().strip("*_`").strip()
        if not cleaned or cleaned.rstrip(":").strip().lower() in _PLAN_PRESCRIBED_HEADINGS:
            continue
        if _leading_verb(cleaned) in _PLAN_EXPLORATION_VERBS:
            offenders.append(cleaned)
    return offenders


def _check_plan_shape(
    agent: Any, tool_name: str, arguments: dict, execution_context: dict[str, Any] | None
) -> str | None:
    """Refuse a plan document whose axes are exploration steps.

    Verification-tier guard, never enforcement-tiered: it holds a plan that has not
    been paid for, and the refusal costs one turn because the message names the
    offending axes verbatim — a stated exit, not a wall. The plan is refused BEFORE
    it is recorded, so there is no document to clear afterwards.

    Targeted by capability and arg-role, never by tool name: the ``plan_document``
    role reaches the prose plan and never the checklist (``plan_steps``), where
    "validate the solver" is a legitimate step.
    """
    if not has_cap(tool_name, TASK_PLANNING, agent.tool_caps):
        return None
    body_args = arg_role(tool_name, "plan_document", agent.tool_caps)
    if not body_args:
        return None
    text = str(arguments.get(body_args[0]) or "")
    if not text.strip():
        return None
    try:
        offenders = _exploration_axes(text)
    except Exception:
        return None  # fail open: a malformed plan is the model's problem, not a block
    if not offenders:
        return None
    named = "; ".join(f"'{axis}'" for axis in offenders[:4])
    return agent._json_error_payload(
        f"Plan not recorded: {len(offenders)} of its axes are exploration, not change — {named}.",
        hint=(
            "Every axis of a plan is a change to make. Finding out what has to change "
            "is what the exploration phase is for, and a plan that opens with it is a "
            "guess about what that exploration will return. Either do that work now "
            "with the read-only tools and write the plan over what you find, or — if "
            "you already know the answer — restate the axis as the change it leads to. "
            "An assumption you could not settle belongs in the plan's prose, named as "
            "an assumption; it is never an axis."
        ),
        tool=tool_name,
    )

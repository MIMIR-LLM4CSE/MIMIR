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

from ...context.capabilities import (
    CLUSTER_SUBMIT, CODE_EXEC, EDIT, EXTERNAL_FETCH, PLAN_BLOCKED, READ,
    arg_role, has_cap,
)


# ── external fetch ────────────────────────────────────────────────────────────

def _check_external_fetch(
    agent: Any, tool_name: str, execution_context: dict[str, Any] | None
) -> str | None:
    """Block an EXTERNAL_FETCH tool until some local discovery has happened.

    Capability-driven replacement for the former hardcoded ``github_*`` name block.
    The guard runs at call time (the tool stays visible in the prompt, keeping the
    per-query tool list stable) and rejects a reach-outside-the-workspace call when
    the agent has not yet searched, inspected, or read anything locally. Returns a
    structured violation payload, or ``None`` when the call is allowed.
    """
    if execution_context is None:
        return None
    if not has_cap(tool_name, EXTERNAL_FETCH, agent.tool_caps):
        return None
    local_discovery_done = (
        execution_context.get("searched")
        or bool(execution_context.get("inspected_dirs"))
        or bool(execution_context.get("read_files"))
    )
    if local_discovery_done:
        return None
    return agent._json_error_payload(
        f"External fetch '{tool_name}' blocked: gather local context first.",
        hint=(
            "Search, inspect, or read the local workspace before reaching outside it "
            "(e.g. grep / list_directory / read_file_lines). Retry this external call once "
            "local discovery has been done."
        ),
        tool=tool_name,
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
    about is exactly what would otherwise be refused. Capability-scoped to CODE_EXEC
    and driven off the shared segmenter, so no tool name or shell keyword is spelled
    out here. Fail-open on an unparseable command: the server still validates and
    confines every accepted call independently.
    """
    if not has_cap(tool_name, CODE_EXEC, getattr(agent, "tool_caps", None)):
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

    ``cluster_submit_warned`` is kept, but only to shorten the message after the first
    hold: repeating the full rationale on every attempt is noise, while dropping the
    guard is the thing being fixed.

    Returns a structured violation payload, or ``None`` when the call is allowed.
    """
    if execution_context is None:
        return None
    if not has_cap(tool_name, CLUSTER_SUBMIT, agent.tool_caps):
        return None
    if execution_context.get("validated_files"):
        return None
    if execution_context.get("cluster_submit_warned"):
        return agent._json_error_payload(
            f"Cluster submission '{tool_name}' still held: nothing has been validated locally.",
            hint=(
                "Run a local check that passes first (a compile, an import check, a quick "
                "smoke run). Retrying alone will not clear this."
            ),
            tool=tool_name,
        )
    execution_context["cluster_submit_warned"] = True
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
# reported as a win. Scoped to CODE_EXEC tools; blocks only calls that *execute* the
# proxy (command position), never read-only inspection of its source. Locked and
# non-tiered — a correctness boundary, not guidance. Fail-open on internal error.

# Leading tokens that wrap another command; we look past them (and their flags /
# ``VAR=val`` assignments) to find the program actually executed.
_EXEC_WRAPPERS = frozenset({"env", "time", "nohup", "stdbuf", "nice", "ionice", "xargs", "srun"})
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

    Fast-path abstains for any non-CODE_EXEC tool (no disk touched). Otherwise, if
    an optimization session is initialized, blocks the call only when one of its
    string arguments *executes* the proxy source/executable (in command position).
    Read-only inspection of the source (``cat``/``grep``) never matches. Fail-open.

    Note: this covers a bash ``python proxy.py`` / ``./proxy``. Executing the proxy
    by pasting its body inline into a fresh script is a narrower, documented residual
    gap; the reserved-metrics guard still prevents forging an accepted result there.
    """
    if not has_cap(tool_name, CODE_EXEC, getattr(agent, "tool_caps", None)):
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

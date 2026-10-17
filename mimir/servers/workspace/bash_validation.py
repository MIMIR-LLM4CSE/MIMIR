"""What a command must pass before bash_run runs it: every segment parsed, its program
on the allow-list, its paths and redirections inside the workspace, its module and
environment-manager arguments well formed. A refusal says why and what to do instead."""
from __future__ import annotations

import os
import re
import shlex
from .._shared.responses import err, ok
from .._shared.shell_paths import (
    CLUSTER_SUBMIT_COMMANDS,
    DESTRUCTIVE_COMMANDS,
    ENV_MANAGER_COMMANDS as _ENV_MANAGER_COMMANDS,
    READONLY_NESTED_COMMANDS,
    SHELL_INTERPRETERS,
    ShellParseError as _ShellParseError,
    cd_destination as _cd_destination,
    expansion_operands,
    normalize_path_arg as _normalize_path_arg,
    denied_command,
    parse_segments as _parse_segments,
    segment_path_operands,
    unwrap_argv,
)

WORKSPACE_ROOT = os.path.realpath(
    os.path.abspath(os.environ.get("MCP_FILES_ROOT", os.getcwd()))
)


# Commands run with the user's real home as HOME (see _safe_env, which rebuilds the
# env from scratch); the validator needs the same value to know where a bare 'cd' lands.
REAL_HOME = os.path.realpath(os.path.abspath(os.path.expanduser("~")))


# Sub-commands that run an arbitrary nested command, refused for the same reason a
# shell interpreter is: this validator would never see what they execute. Everything
# else an environment manager does — install, uninstall, create, remove, config — is
# reviewable from the command line and runs under the approval prompt.
_ENV_MANAGER_NESTING_SUBCOMMANDS = {"run", "execute"}


def _validate_env_manager_args(argv: list[str]) -> dict | None:
    """Keep an environment manager from nesting a command (see the set above)."""
    positionals = [a for a in argv[1:] if not a.startswith("-")]
    sub = positionals[0] if positionals else None
    if sub is None:
        return err(
            f"'{argv[0]}' needs a sub-command.",
            hint="Name what to do, e.g. 'install', 'list', 'freeze'.",
        )
    if sub in _ENV_MANAGER_NESTING_SUBCOMMANDS:
        return err(
            f"'{argv[0]} {sub}' is not available here.",
            hint=f"'{argv[0]} {sub}' executes a nested command this validator never "
                 f"sees. Write that command out directly instead; to run it in another "
                 f"environment, invoke that environment's interpreter by path.",
        )
    return None


# A safe 'module' argument: a module name (e.g. 'cuda/12.2', 'gcc/11.3.0') or a
# benign flag (e.g. '-t', '--terse'). Tokenization already neutralizes shell
# metacharacters; this is a second, narrower gate on what reaches Lmod.
_MODULE_ARG_RE = re.compile(
    r"^(?:-{1,2}[A-Za-z][A-Za-z-]*|[A-Za-z0-9][A-Za-z0-9._/+-]*)$"
)


# Why a command could not be segmented → what the agent should do about it. The
# refusal itself (and its wording) comes from shell_paths.parse_segments, which the
# client shares; only the remedy is server-side, since only this side has an
# denylist to point at.
_PARSE_HINTS = {
    "empty": "Provide at least one command.",
    "separator": "Check for a leading or doubled ';', '&&', '||' or '|'.",
    "substitution": "Remove $(...), backticks, ${...}, or process substitution. "
                    "Double quotes do not protect them. To run a script that "
                    "contains them, write it to a file with the file-write tool "
                    "and run that file.",
    "operator": "Chaining (';', '&&', '||', '|') and redirection ('> out.txt', "
                "'2>&1') are allowed; backgrounding ('&') and subshells are not.",
    "heredoc": "A heredoc body is not a command this can validate. Pass the text as "
               "a quoted argument, or write it with the file-write tool, then "
               "redirect from it ('< input.txt').",
    "redirect": "Give the redirection a target file ('> out.txt') or an fd "
                "('2>&1').",
}


# The subset of _EXEC_COMMANDS that honours \write18 — the escape hatch that lets a
# .tex document run an arbitrary shell command at compile time. Every flag that
# re-enables it is denylisted (see _TEX_FORBIDDEN_FLAGS) and _safe_env pins the
# kpathsea knobs. ('bibtex'/'biber'/'makeindex'/'dvips' have no such hatch.)
_TEX_COMMANDS = {
    "pdflatex", "latex", "xelatex", "lualatex", "pdftex", "tex", "latexmk",
}


# Shell-escape / write18 re-enablement, in every spelling the engines accept.
_TEX_FORBIDDEN_FLAGS = {
    "-shell-escape", "--shell-escape", "-enable-write18", "--enable-write18",
    "-enable-pipes", "--enable-pipes", "-escape-shell", "--escape-shell",
}


# Per-command denylist of flags whose write or exec target sits where the path
# extraction cannot see it — a flag *value* (`find -fprint FILE`) or a nested command
# (`find -exec`, `rg --pre`). Matched as an exact token or a glued '=' form. Keyed by
# argv[0]. A write flag whose target the guard *can* see is confined instead of denied
# (``shell_paths.WRITE_VALUE_FLAGS_BY_CMD``: `gcc -o`, `sort -o`, …).
_FORBIDDEN_ARG_TOKENS_BY_CMD = {
    # -exec/-execdir are NOT here: the parser lifts their payload into its own nested
    # segment, judged on the nested command's own head (see validate_command).
    # -ok/-okdir stay — they prompt on a tty the agent does not have; -delete
    # does not, and its operand is confined like any other.
    "find": {"-ok", "-okdir",
             "-fprint", "-fprint0", "-fprintf", "-fls", "--in-place"},
    "rg":   {"--pre", "--pre-glob", "--hostname-bin", "--search-zip", "-f", "--file"},
    # '-f/--file' reads patterns from a file — a read whose operand sits in flag-value
    # position, exactly where the pattern-skipping rule stops looking.
    "grep": {"-f", "--file"},
    **{cmd: _TEX_FORBIDDEN_FLAGS for cmd in _TEX_COMMANDS},
}


# Why each denied group is denied, and what to do instead. A refusal that names no
# route is a dead end, and the model retries variants of it.
_DENIAL_HINTS = {
    "interpreter": (
        "'{name}' runs a nested command this validator never sees, so allowing it "
        "would void every check that follows — path confinement included. Write the "
        "command out directly: chaining (';', '&&', '||', '|') and redirection are "
        "supported, so anything you would put in a script can be one chain. To run a "
        "script that exists in the workspace, invoke it by path ('./build.sh'), with "
        "'chmod +x ./build.sh' first if needed. To set up an environment for the rest "
        "of a chain, use 'source env.sh'."
    ),
    "destructive": (
        "'{name}' destroys data outside anything this session can review or undo. "
        "There is no variant of it that is available — report the limitation. To "
        "remove files from the workspace, 'rm' is available and its operands are "
        "checked."
    ),
    "cluster": (
        "'{name}' submits or cancels a Slurm job, which goes through the cluster tools "
        "instead: submitting returns a job handle this session tracks, resuming on its "
        "own when the job ends, and cancelling takes one job of yours at a time. A job "
        "submitted from a shell is tracked by nothing. 'squeue'/'sinfo'/'sacct' are "
        "available here for inspection."
    ),
}


def _denial_kind(name: str) -> str:
    if name in SHELL_INTERPRETERS:
        return "interpreter"
    if name in DESTRUCTIVE_COMMANDS:
        return "destructive"
    assert name in CLUSTER_SUBMIT_COMMANDS
    return "cluster"


def module_preamble() -> str:
    """A wrapper prefix that defines the Lmod 'module' function in a --norc shell.

    Returns a ``source <init>; `` string when this host has Lmod's bash init,
    else ``""``. This is prepended by the server to the *validated* user command
    before handing it to ``bash -c``; because the server builds it (the user
    never supplies it), its source/eval is not — and need not be — subject to
    the command validator's no-substitution rule.
    """
    home = os.environ.get("MODULESHOME", "")
    if not home:
        return ""
    init = os.path.join(home, "init", "bash")
    if not os.path.isfile(init):
        return ""
    return f"source {shlex.quote(init)} >/dev/null 2>&1; "


def validate_module_args(argv: list[str]) -> dict | None:
    """Check a 'module ...' invocation's argument shapes.

    Every sub-command is available — load, unload, swap and purge all change the
    session's environment and nothing else, and the approval prompt covers that. What
    is checked is the shape of what reaches Lmod, which evaluates modulefiles.
    """
    if len(argv) < 2:
        return err(
            "'module' needs a subcommand.",
            hint="Use e.g. 'module avail', 'module list', or 'module load cuda'.",
        )
    for arg in argv[1:]:
        if not _MODULE_ARG_RE.match(arg):
            return err(
                f"Unsafe argument to module: {arg!r}",
                hint="Pass module names (e.g. 'cuda/12.2') or simple flags only.",
            )
    return None


def is_within_workspace(path: str) -> bool:
    root = WORKSPACE_ROOT
    if path == root or path.startswith(root + os.sep):
        return True
    # Out-of-workspace locations the user approved this session, plus the system's
    # own standing grant (the scratchpad). Both read per-call: the server's env is
    # frozen at spawn, so the shared state dir is the only live channel.
    try:
        from .._shared.approved_roots import approved_roots
        from .._shared.state_paths import standing_roots
        for extra in list(approved_roots()) + standing_roots():
            base = os.path.realpath(os.path.abspath(os.path.expanduser(extra)))
            if path == base or path.startswith(base + os.sep):
                return True
    except Exception:
        pass
    return False


def _validate_redirections(segment, cwd: str) -> dict | None:
    """Confine the files a segment's redirections read from and write to.

    A redirection is a path operand written with an operator instead of a flag:
    ``gcc -o solver.out`` and ``./solver.out > run.log`` both create a file, so both
    are judged the same way — the target is resolved against *cwd* (which a ``cd``
    earlier in the chain has already rebased) and must sit inside the workspace or
    a directory the user approved. Nothing about a redirection reaches outside that,
    including through a symlink: the resolution is ``realpath``.

    fd dups and ``/dev/null`` never get here — they name no file, so
    :func:`parse_segments` drops them.
    """
    for redirection in segment.redirections:
        target = redirection.target
        if "$" in target or "`" in target:
            # Same divergence as any other expanded path (see _validate_path_args),
            # with a write on the other end of it.
            return err(
                f"A redirection target built from a shell expansion cannot be "
                f"checked: {target}",
                hint="Expansion happens in the shell that runs the command, not in "
                     "the one checking it, so the file checked would not be the "
                     "file written. Write the path out literally.",
            )
        resolved = _normalize_path_arg(target, cwd)
        if resolved is None:
            continue
        if not is_within_workspace(resolved):
            verb = "written" if redirection.writes else "read"
            return err(
                f"Path outside workspace is not approved: {resolved}",
                hint=f"A redirection target is a file like any other — a file "
                     f"{verb} outside the workspace needs the user's approval. Ask "
                     f"for it, or redirect to a path inside the workspace root.",
            )
    return None


def _resolve_cd_target(argv: list[str], cwd: str, call_cwd: str) -> tuple[str | None, dict | None]:
    """Resolve where a ``cd`` segment moves the shell, or reject it.

    ``cd``'s target is a path operand, confined like any other —
    and, crucially, the directory it lands in becomes the base against which every
    *later* segment's paths are validated. Without that propagation the confinement
    of the rest of the chain is validated against a directory the shell has already
    left, and ``cd /etc && cat passwd`` reads outside the sandbox with each segment
    looking individually fine.

    A bare ``cd`` or ``cd ~`` goes to the user's real home (HOME is inherited, not
    pinned — see _safe_env), so it is judged like any other destination outside the
    workspace: refused unless the user approved it. ``cd -`` resets to the workspace
    root; see :func:`shell_paths.cd_destination` for why that is safe.

    Moving *outside* the workspace is not forbidden, it is **user-approved**: the
    client surfaces the destination through the ordinary out-of-workspace prompt and
    a grant lands in the sidecar that ``is_within_workspace`` reads per call. So the
    check below passes for a directory the user approved this session and refuses one
    they did not — the refusal says so, because "ask the user" is an action the agent
    can actually take, unlike a flat denial.
    """
    resolved = _cd_destination(argv, cwd, call_cwd)
    if not is_within_workspace(resolved):
        # The *resolved* destination, not the token written, so the refusal and the
        # approval prompt name the same path ('..' and '/etc' do not).
        return None, err(
            f"Path outside workspace is not approved: {resolved}",
            hint="Moving outside the workspace needs the user's approval for that "
                 "directory. Ask for it, or stay inside the workspace root.",
        )
    return resolved, None


def _validate_path_args(argv: list[str], cwd: str) -> dict | None:
    """Confine every path a command names to the workspace (or an approved root).

    The operand extraction is ``shell_paths.segment_path_operands`` — the same
    routine the client's out-of-workspace gate walks to decide what to prompt for,
    so a path the user is asked about is exactly a path this guard would refuse.
    A workspace-local binary (``./solver.out``) is covered too: it is an execution
    like any other, and ``./solver.out /etc/secrets`` would otherwise read out.
    """
    expanded = expansion_operands(argv)
    if expanded:
        return err(
            f"A path built from a shell expansion cannot be checked: {expanded[0]}",
            hint="Expansion happens in the shell that runs the command, not in the "
                 "one checking it, so the path checked would not be the path read "
                 "(`module load cuda && cat $CUDA_HOME/f` resolves differently in "
                 "each). Write the path out literally. Expansions inside flags "
                 "(`-I$CUDA_HOME/include`) are unaffected.",
        )

    for resolved in segment_path_operands(argv, cwd):
        if not is_within_workspace(resolved):
            return err(
                f"Path outside workspace is not approved: {resolved}",
                hint="Paths outside the workspace need the user's approval. Ask for "
                     "it, or use a path inside the workspace root.",
            )
    return None


def validate_command(command: str, cwd: str) -> dict:
    """Decide whether this command may run at all, before any shell is opened.

    Segmentation comes from shell_paths.parse_segments, shared with the client's gate.
    What is left here is the *policy*: the denylist, per-command flag rules and path
    confinement. Everything else runs — the approval prompt is the barrier.

    Every rejection carries ``refused``. The client cannot otherwise tell one apart from
    a command that ran and exited non-zero, and the two mean opposite things: a red exit
    is a finding about the change, while a rejection never reached a shell and is
    evidence of nothing. Read as a plain error, a refused heredoc was entered in the run
    ledger as red — charging the repair budget and printing its whole body into the
    user's chat as an unresolved run.
    """
    return _refused(_validate(command, cwd))


def _refused(result: dict) -> dict:
    """Stamp a validation answer that rejected the command."""
    return result if result["status"] == "ok" else {**result, "refused": True}


def _validate(command: str, cwd: str) -> dict:
    try:
        segments = _parse_segments(command)
    except _ShellParseError as e:
        return err(e.message, hint=_PARSE_HINTS.get(e.reason, ""))

    parsed = []
    # Base for path confinement; a 'cd' segment moves it (see _resolve_cd_target).
    seg_cwd = cwd
    for segment in segments:
        # A wrapper's argument list is another command ('timeout 60 pytest -q'), so
        # unwrap before anything reads the head — otherwise every check below judges
        # the wrapper and none of them judges what actually runs.
        argv, _wrappers = unwrap_argv(segment.argv)
        # Matched by basename too: '/bin/sh' is the 'sh' that is refused, and allowing
        # it by path would reinstate the very bypass.
        denied = denied_command(argv[0])
        if denied is not None:
            kind = _denial_kind(denied)
            return err(
                f"Command '{denied}' is not available here.",
                hint=_DENIAL_HINTS[kind].format(name=denied),
                denial=kind,
            )

        if argv[0] == "module":
            module_err = validate_module_args(argv)
            if module_err is not None:
                return module_err

        if argv[0] in _ENV_MANAGER_COMMANDS:
            env_err = _validate_env_manager_args(argv)
            if env_err is not None:
                return env_err

        if argv[0] == "cd":
            # A redirection on a 'cd' is pointless but harmless; check it against
            # the directory the shell is in *before* the move, as bash does.
            redir_err = _validate_redirections(segment, seg_cwd)
            if redir_err is not None:
                return redir_err
            # An expanded destination has the same validate-vs-execute divergence as
            # any other path, and worse consequences: it rebases the whole chain.
            cd_expansion_err = _validate_path_args(argv, seg_cwd)
            if cd_expansion_err is not None:
                return cd_expansion_err
            new_cwd, cd_err = _resolve_cd_target(argv, seg_cwd, cwd)
            if cd_err is not None:
                return cd_err
            seg_cwd = new_cwd
            parsed.append(argv)
            continue

        # A nested command the parser lifted out (`find … -exec CMD {} \;`) is a real
        # command: allowed only if its head is read-only. Judged here rather than by
        # the flag denylist below, which would reject it on the `-exec` token alone.
        if segment.nested:
            if argv[0] not in READONLY_NESTED_COMMANDS:
                return err(
                    f"A nested command that is not read-only was detected: {argv[0]}",
                    hint="A nested command's operands include the '{}' placeholder, "
                         "whose expansion cannot be resolved here, so a nested write "
                         "or execution targets paths that cannot be checked. Read-only "
                         "nested commands (grep, ls, cat, wc, …) are supported. To act "
                         "on the matches, list them first and pass the paths "
                         "explicitly in a second command.",
                )
            path_err = _validate_path_args(argv, seg_cwd)
            if path_err is not None:
                return path_err
            parsed.append(argv)
            continue

        forbidden_flags = _FORBIDDEN_ARG_TOKENS_BY_CMD.get(argv[0], set())
        if forbidden_flags and any(
            arg in forbidden_flags or any(arg.startswith(tok + "=") for tok in forbidden_flags)
            for arg in argv[1:]
        ):
            return err(
                "A flag whose write or exec target cannot be checked was detected.",
                hint="This flag hides its destination (or nests a command) where the "
                     "path check cannot see it. Writing a file is fine by "
                     "redirection — '<command> > out.txt' — whose target is "
                     "checked. A nested read-only command is supported as "
                     "'find … -exec grep -l PATTERN {} \\;'.",
            )

        path_err = _validate_path_args(argv, seg_cwd)
        if path_err is not None:
            return path_err

        redir_err = _validate_redirections(segment, seg_cwd)
        if redir_err is not None:
            return redir_err

        parsed.append(argv)

    return ok({"segments": parsed})

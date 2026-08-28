"""
MCP Bash Server
===============
Controlled bash command execution: read-only workspace inspection, plus the
direct toolchain for compiling, running, validating, and testing code.

WHAT THIS VALIDATION IS, AND IS NOT
-----------------------------------
Read this before trusting anything below, because two mechanisms are in play and
only one of them lives in this file.

The user's real control is the **approval prompt**, which the client applies to
every build/exec command — inside the workspace as much as outside. ``python
solver.py``, ``make``, ``./a.out`` and ``pdflatex main.tex`` all stop and ask.
Nothing here runs unasked (except under the headless runner, which auto-approves).

The validation *in this file* is the narrower half: it governs the command the
agent writes — which program runs, and which paths it names. It says nothing about
what that program does once running. ``python``, ``make``, ``gcc`` and a
workspace-local ``./a.out`` execute with the full privileges of the account running
MIMIR, so a script the agent wrote can read any file that account can read, write
anywhere it can write, and open a socket — none of which this module sees. ``make``
runs a workspace Makefile, arbitrary shell by construction. ``python -c
"open('/etc/passwd')"`` passes every check here, because the path never appears as
an argument.

So the user controls *whether* something runs; nothing here controls *what it does*
once it does. Constraining the latter needs isolation at the process level
(namespaces, seccomp, a container, bubblewrap) — a different mechanism, not more
rules here, and not implemented.

Read "confined" throughout this file with that scope in mind; SERVERS_DETAILED.md
carries the same note with the bounds on what an "always" approval grants.

WHAT IS REFUSED, AND WHY
------------------------
Anything not named below runs. The three groups here are refused because no approval
prompt could make them reviewable, and each rejection names the route that replaces it
— a refusal with no route is a dead end the model retries variants of.

=============  ==================================================================
group          commands
=============  ==================================================================
interpreter    bash sh zsh ksh dash csh tcsh fish eval exec . command sudo su doas
destructive    shred dd mkfs fdisk parted swapoff mkswap
cluster        sbatch salloc scancel
=============  ==================================================================

An interpreter runs a nested command this validator never sees, which would void
segmentation and with it path confinement — so it stays refused in every spelling,
including by absolute path and behind a wrapper (``timeout 5 /bin/sh -c x`` is refused
on ``sh``). ``source`` is the deliberate exception: its operand is a *file*, confined
like any other path, and it is the only way to set the environment the rest of a chain
runs in (``source venv/bin/activate && pytest``). The ``.`` spelling stays refused.
Job submission has one route, the typed cluster tools, which return a handle this
session tracks; ``squeue``/``sinfo``/``sacct`` run here freely. Deletion (``rm``) is
*not* refused — destructive but reviewable, so it runs under the approval prompt with
its operands confined, like any other write.

Wrappers whose argument list is another command — ``timeout``, ``nohup``, ``env``,
``xargs``, ``stdbuf``, ``nice``, ``ionice``, ``srun``, ``mpirun``, ``mpiexec`` — are
unwrapped before validation (``shell_paths.unwrap_argv``), so ``timeout 60 pytest -q``
is validated and classified as the pytest run it is.

WHAT A COMMAND'S *CATEGORY* STILL DECIDES
-----------------------------------------
The taxonomy in ``_shared/shell_paths.py`` is no longer a gate; it is what the client's
classifier reads to decide approval and plan-mode availability. Read/search/inspect
commands (``cat``, ``grep``, ``ls``, ``pip list``, ``module avail``, ``cd``) are
read-only: they run unattended and are available while planning. Everything else —
writes, execution, installs, and every head the taxonomy does not place — is
approval-gated and blocked while planning.

Per-call shifts: ``sed -i`` / ``sort -o FILE`` and a ``> file`` redirection turn a
side-effect-free command into a **write**; ``< file`` and fd redirection change nothing;
for the env managers the sub-command decides (query vs mutation), and an unknown one is
assumed to mutate. In a chain, one non-read-only segment makes the whole call
non-read-only.

This server is intentionally restricted:
- only runs inside the workspace root
- it is the primary surface for compile/run/test/validate — invoke the toolchain
  directly with the exact flags the task needs
- command chaining is allowed (``;``, ``&&``, ``||``, ``|``) since real tasks
  often need several commands in one call; each command between separators is
  tokenized and validated independently
- redirection is allowed — fd forms (``2>&1``, ``1>&2``, ``2>/dev/null``) and
  named files (``> run.log``, ``>> run.log``, ``< input.txt``). A redirection
  target is a path operand written with an operator instead of a flag, so it is
  confined exactly like one (``ls > /tmp/out.txt`` is refused for the same reason
  ``cat /etc/passwd`` is); a heredoc (``<<EOF``) is not, its body being no argv
- multi-line commands are accepted: an *unquoted* newline is a separator like
  ``;``, so it is normalized to one before tokenizing and each line is validated
  as its own command. Newlines inside quotes (a multi-line ``python3 -c "..."``
  body) are data and stay untouched
- still rejected: backgrounding (``&``), command/process substitution
  (``$(...)``, backticks, ``<(...)``, ``${...}``) and subshells
- runs ``bash -c`` only after validation, so globbing (``*.py``) and ``$VAR``
  expansion still work
- the environment managers (``pip``, ``conda``, ``mamba``) run their whole surface,
  install and uninstall alike, minus ``conda run``/``mamba run``, which nest an
  unvalidated command. An install writes into the target interpreter's
  site-packages — outside the workspace by construction, so the approval prompt,
  not a path check, is what governs it
- ``module`` (HPC Lmod) is supported: it is a shell *function*, so the server
  defines it by sourcing Lmod's init in the wrapper it builds around the
  validated command. This means ``module load cuda && nvcc ...`` works in a
  single call; a load does not persist to the next call (fresh subprocess)

Every call starts at the workspace root — there is no working-directory argument.
``cd`` holds for the rest of a single call (``cd sub && pytest
t.py``), but each call is a fresh subprocess so it does not carry over. One way to
say "where", not two: a second mechanism was only ever a second thing to keep
confined, and the root is the one base every path in the call is judged against.

Critical hardening applied:
- one segmenter, shared with the client: ``shell_paths.parse_segments`` decides
  where a command ends and which tokens are a redirection rather than an argument,
  so the guard that *confines* a path and the gate that *prompts* for it can never
  read the same command two ways (see that module's docstring)
- treats an unquoted newline as the command separator it is, so every line of a
  multi-line command is validated on its own (never folded into the argv of the
  line above)
- resolves every path operand with realpath() to prevent symlink escapes
- per-command denylist of flags whose write/exec target the operand extraction
  cannot see (find ``-exec``/``-fprint``, rg ``--pre``, grep ``-f``)
- confines the path arguments of every command that takes a file operand —
  readers, writers AND the build/exec toolchain — plus every redirection target,
  to the workspace root or a path the user approved this session
- threads the current directory through a chain, so a ``cd`` rebases the
  confinement of the segments after it
- runs bash with --noprofile --norc and a minimal env
"""

import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_READONLY, CODE_EXEC, JUDGE, RECOVERABLE
from responses import err, ok
from shell_paths import (
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

mcp = FastMCP(
    "BashServer",
    debug=False,
    log_level="ERROR",
)

_WORKSPACE_ROOT = os.path.realpath(
    os.path.abspath(os.environ.get("MCP_FILES_ROOT", os.getcwd()))
)
_MAX_OUTPUT = 64 * 1024
# Sized for the work this shell actually does now that it compiles and runs: a default
# that survives an ordinary build or test suite, and a ceiling past which a run belongs
# to the background-job route rather than to a call that blocks the turn.
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300

# Commands run with the user's real home as HOME (see _safe_env, which rebuilds the
# env from scratch); the validator needs the same value to know where a bare 'cd' lands.
_REAL_HOME = os.path.realpath(os.path.abspath(os.path.expanduser("~")))

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

# Env vars Lmod's init/bash needs to locate and evaluate modulefiles. Passed
# through (when present) on top of the otherwise-minimal subprocess env so that
# 'module avail'/'module load' actually resolve the site's module tree.
_MODULE_ENV_PASSTHROUGH = (
    "MODULESHOME", "MODULEPATH", "MODULEPATH_ROOT",
    "LMOD_CMD", "LMOD_DIR", "LMOD_PKG", "LMOD_ROOT",
    "LMOD_SYSTEM_DEFAULT_MODULES", "LMOD_sys", "LMOD_arch", "LMOD_SYSHOST",
)

# Why a command could not be segmented → what the agent should do about it. The
# refusal itself (and its wording) comes from shell_paths.parse_segments, which the
# client shares; only the remedy is server-side, since only this side has an
# denylist to point at.
_PARSE_HINTS = {
    "empty": "Provide at least one command.",
    "separator": "Check for a leading or doubled ';', '&&', '||' or '|'.",
    "substitution": "Remove $(...), backticks, ${...}, or process substitution.",
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
    # segment, judged on the nested command's own head (see _validate_command).
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
        "'{name}' submits a job, which goes through the cluster submission tool "
        "instead: it returns a job handle this session tracks, resuming on its own "
        "when the job ends. A job submitted from a shell is tracked by nothing. "
        "'squeue'/'sinfo'/'sacct' are available here for inspection."
    ),
}


def _denial_kind(name: str) -> str:
    if name in SHELL_INTERPRETERS:
        return "interpreter"
    if name in DESTRUCTIVE_COMMANDS:
        return "destructive"
    assert name in CLUSTER_SUBMIT_COMMANDS
    return "cluster"


# Commands whose exit status 1 means "nothing found" rather than "failed" — a
# legitimate answer the model would otherwise re-run unchanged. A clean no-match
# (rc 1, empty stdout) is reported as success; a real failure exits 2 and stays an error.
_NO_MATCH_COMMANDS = {"grep", "rg", "which"}


def _safe_env(cwd: str) -> dict:
    """Minimal environment for subprocess execution.

    PATH is prefixed with the directory of the interpreter this server runs under
    (``sys.executable``). A bare ``python``/``python3`` in a bash_run therefore
    resolves to the *same* environment MIMIR itself uses — the one that has the
    project's dependencies — rather than to whatever ``python3`` happens to sit
    first on the ambient PATH (which may be a different env, or even a different
    CPU architecture on a mixed x86/ARM host, and would fail or lack the deps).
    """
    interp_dir = os.path.dirname(os.path.abspath(sys.executable))
    ambient_path = os.environ.get("PATH", "")
    path = interp_dir + os.pathsep + ambient_path if interp_dir else ambient_path
    env = {
        "PATH": path,
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        # Inherited, not pinned to the workspace: confinement comes from the operand
        # path check, and pinning only detached tools from the user's own config and
        # caches (~/.local site-packages, ~/.condarc, ~/texmf). Startup files are kept
        # out by 'bash --noprofile --norc' plus BASH_ENV below, which is what that
        # concern actually needed.
        "HOME": _REAL_HOME,
        # Prevent non-interactive bash from loading an arbitrary startup file.
        "BASH_ENV": "",
        "PWD": cwd,
        # kpathsea reads these from the env, pinning the TeX sandbox regardless of the
        # site's texmf.cnf: no \write18, and 'paranoid' file access. Backstop for the
        # shell-escape flags already denylisted at validation.
        "shell_escape": "f",
        "openout_any": "p",
        "openin_any": "p",
    }
    # Let Lmod find the site's modulefiles when a 'module' command is in play.
    # These are config/path vars only — they do not themselves run code.
    for var in _MODULE_ENV_PASSTHROUGH:
        val = os.environ.get(var)
        if val:
            env[var] = val
    return env


def _module_preamble() -> str:
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


def _validate_module_args(argv: list[str]) -> dict | None:
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


def _is_within_workspace(path: str) -> bool:
    root = _WORKSPACE_ROOT
    if path == root or path.startswith(root + os.sep):
        return True
    # Out-of-workspace locations the user approved this session, plus the system's
    # own standing grant (the scratchpad). Both read per-call: the server's env is
    # frozen at spawn, so the shared state dir is the only live channel.
    try:
        from approved_roots import approved_roots
        from state_paths import standing_roots
        for extra in list(approved_roots()) + standing_roots():
            base = os.path.realpath(os.path.abspath(os.path.expanduser(extra)))
            if path == base or path.startswith(base + os.sep):
                return True
    except Exception:
        pass
    return False


def _classify_runtime_error(stderr: str) -> str | None:
    text = stderr.lower()
    if any(k in text for k in [
        "modulenotfounderror",
        "importerror",
        "no module named",
        "command not found",
        "cannot open shared object file",
        "libstdc++",
        "glibc",
        "undefined symbol",
        "qt.qpa",
        "matplotlib",
        "backend",
        "thread",
    ]):
        return "environment_mismatch_likely"
    return None


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
        if not _is_within_workspace(resolved):
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
    a grant lands in the sidecar that ``_is_within_workspace`` reads per call. So the
    check below passes for a directory the user approved this session and refuses one
    they did not — the refusal says so, because "ask the user" is an action the agent
    can actually take, unlike a flat denial.
    """
    resolved = _cd_destination(argv, cwd, call_cwd)
    if not _is_within_workspace(resolved):
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
        if not _is_within_workspace(resolved):
            return err(
                f"Path outside workspace is not approved: {resolved}",
                hint="Paths outside the workspace need the user's approval. Ask for "
                     "it, or use a path inside the workspace root.",
            )
    return None


def _validate_command(command: str, cwd: str) -> dict:
    # Segmentation comes from shell_paths.parse_segments, shared with the client's
    # gate. What is left here is the *policy*: the denylist, per-command flag rules and
    # path confinement. Everything else runs — the approval prompt is the barrier.
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
            module_err = _validate_module_args(argv)
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


def _is_clean_no_match(segments: list[list[str]] | None, returncode: int, stdout: str) -> bool:
    """True when a non-zero exit is really an empty *result*, not a failure.

    Bash reports the exit status of the last command in the chain, so only that
    segment's leading command decides. See :data:`_NO_MATCH_COMMANDS`.

    ``grep``/``rg`` say "no match" with exactly 1 (2 is a real error, e.g. a bad
    pattern). ``which`` instead returns *how many* of its arguments it could not
    find, so the standard multi-name probe — ``which latex pdflatex xelatex`` —
    exits 4 with nothing found, and testing for 1 reported the most conclusive
    answer there is as a failure, under a hint telling the agent to re-read stderr
    for a problem that does not exist.
    """
    if returncode < 1 or stdout.strip():
        return False
    if not segments or not segments[-1]:
        return False
    head = segments[-1][0]
    if head == "which":
        return True
    return returncode == 1 and head in _NO_MATCH_COMMANDS


def _run(
    command: str,
    cwd: str,
    timeout: int,
    preamble: str = "",
    segments: list[list[str]] | None = None,
) -> dict:
    try:
        result = subprocess.run(
            ["bash", "--noprofile", "--norc", "-c", preamble + command],
            cwd=cwd,
            env=_safe_env(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )

        stdout = result.stdout[:_MAX_OUTPUT]
        stderr = result.stderr[:_MAX_OUTPUT]

        if result.returncode == 0:
            payload = ok({
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": cwd,
            })
        elif _is_clean_no_match(segments, result.returncode, stdout):
            payload = ok({
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "cwd": cwd,
                "matches": 0,
                "note": "The command ran correctly and found nothing. This is a "
                        "conclusive negative answer, not a failure — treat the "
                        "result as 'absent' and do not re-run the command.",
            })
        else:
            classification = _classify_runtime_error(stderr)

            payload = err(
                "Command returned a non-zero exit code.",
                returncode=result.returncode,
                stdout=stdout,
                stderr=stderr,
                cwd=cwd,
            )

            if classification:
                payload["diagnostic"] = classification
                payload["hint"] = (
                    "The failure appears to be due to the runtime environment "
                    "(missing or incompatible dependencies), not shell syntax. "
                    "Consider checking the Python environment or loaded modules."
                )
            else:
                payload["hint"] = (
                    "Read stderr before doing anything else: a non-zero exit is "
                    "often a real finding (a failing test, an absent file, a "
                    "compile error) rather than a malformed command. Act on what "
                    "stderr says; do not re-run the same command unchanged."
                )

        if len(result.stdout) > _MAX_OUTPUT or len(result.stderr) > _MAX_OUTPUT:
            payload["truncated"] = True

        return payload

    except subprocess.TimeoutExpired:
        return err(
            f"Command timed out after {timeout}s.",
            hint=f"Raise the call's own 'timeout' (up to {_MAX_TIMEOUT}s) if the work "
                 f"genuinely takes that long, or narrow the scope. Past that ceiling a "
                 f"run does not belong in a call that blocks the turn: submit it as a "
                 f"background job, which returns a handle this session tracks and "
                 f"resumes on.",
            cwd=cwd,
        )
    except Exception as e:
        return err(str(e), cwd=cwd)


@mcp.tool(**tool_caps(
    caps=[PLAN_READONLY, CODE_EXEC], reversibility=RECOVERABLE, non_batch=True,
    fallbacks=['read_file_lines'],
    scope={"args": ["command"], "kind": "command_prefix"},
    risk_note="runs a shell command in the workspace",
    label="Running shell command",
))
def bash_run(command: str, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """Run a controlled bash command inside the workspace root.

    This is the primary way to search file contents (there is no separate grep tool)
    and the primary way to compile, run, validate and test code — invoke the toolchain
    directly with the exact flags the task needs.

        grep -rn "FastMCP" mimir            # search; rg works too. Always -n: a hit
                                            # without its line number costs a read
        gcc solver.c -O3 -o solver.out -lm && ./solver.out
        python -m pytest tests/test_thing.py -q     # also ruff / mypy / py_compile
        ./solver.out > run.log 2>&1 && tail -20 run.log
        module load cuda && nvcc kernel.cu -o kernel.out
        pip list | grep -i numpy            # then: pip install <pkg>

    Read files with the dedicated read tool, not with `sed -n`/`cat` here: it numbers
    the lines it returns, and the edit tools are addressed by line number. Text that
    arrives through this shell carries no numbers, so acting on it means counting them
    by hand — which is how one edit becomes a series of narrowing re-reads.

    Gating: any command runs. Read-only ones (search, read, inspect, `pip list`,
    `module avail`) run unattended and are available while planning; everything else —
    a write, an execution, an install, a `git` call — is shown to the user for approval
    and is blocked while planning. There is no list of permitted commands to consult:
    reach for the tool the task needs. A refusal names why and what replaces it.

    Syntax that works:
    - chaining (';', '&&', '||', '|'), globbing ('*.py'), and multi-line commands (an
      unquoted newline chains like ';'; a newline inside quotes is literal, so a
      multi-line `python3 -c "..."` body is fine).
    - redirection: '2>&1', '2>/dev/null', '> run.log', '>> run.log', '< input.txt'.
    - 'cd sub && pytest t.py' to work in a subdirectory. Every call starts at the
      workspace root and is its own shell, so a 'cd' or a 'module load' holds for the
      rest of that command only.

    Limits worth knowing before you retry something:
    - every path — file operand, '-o' target, redirection target, 'cd' destination —
      must be inside the workspace, or the user must approve that path. Ask for it
      rather than looking for a way around it.
    - no shell interpreter (bash, sh, eval, sudo): it would run a command nothing
      here can check, and chaining covers what a script would do. Wrappers that take a
      command are fine ('timeout 60 pytest -q', 'env A=B ./run'). No heredoc — pass the
      text as a quoted argument. No backgrounding, substitution or subshell.
    - no job submission (sbatch/salloc): the cluster tools return a handle this session
      tracks. No 'dd'/'shred'. These are absent by design: report the limitation
      instead of trying another spelling.
    - 'sed -i' rewrites in place (prefer the file-edit tools); a bare 'pip install'
      provisions the interpreter MIMIR itself runs under.
    - an empty result from a probe ('which pdflatex') is a conclusive "absent", not an
      error to retry.

    Args:
        command: Shell command string (single command or simple pipeline).
        timeout: Seconds to wait, capped at 300. Raise it for a real build or suite;
            past the cap, submit the run as a background job instead.
    """
    cwd = _WORKSPACE_ROOT

    validation = _validate_command(command, cwd)
    if validation["status"] != "ok":
        return validation

    # If a 'module' command is in the chain, define the Lmod function in the
    # wrapper so 'module load cuda && nvcc ...' works within this single shell.
    needs_modules = any(seg and seg[0] == "module" for seg in validation["segments"])
    preamble = _module_preamble() if needs_modules else ""

    requested_timeout = timeout
    timeout = max(1, min(timeout, _MAX_TIMEOUT))
    result = _run(command, cwd, timeout, preamble=preamble, segments=validation["segments"])
    if timeout != requested_timeout:
        result["timeout_clamped"] = True
        result["requested_timeout"] = requested_timeout
    return result


_VERDICT_VALUES = ("pass", "fail", "unknown", "blocked")


@mcp.tool(**tool_caps(
    caps=[JUDGE],
    arg_roles={
        "verdict": ["verdict"], "verdict_reason": ["reason"], "verdict_scope": ["run"],
    },
    label="Verdict: {verdict}",
))
def report_verdict(verdict: str, reason: str, run: str = "") -> dict:
    """State what a run's output showed. Recommended after any execution you had to read.

    Exit 0 means a program reached its end, never that its answer is right, and nothing
    downstream can read the output for you. Until you report it, the run stands as
    something that happened and nothing more. A run whose output settles nothing, and a
    syntax, lint or type check, need no report — an exit code is the whole finding
    there, and all it establishes: that the file parses and lints, never that the result
    is right.

    Report as soon as you have read the output, in the same step you would move on:

        report_verdict("pass", "l2_rel=3.1e-4 against the analytic solution, under the 1e-3 bound")
        report_verdict("fail", "energy grows from 1.56 to 4.02 — the absorbing layer reflects")
        report_verdict("unknown", "only prints 'Simulation completed.', nothing about correctness")
        report_verdict("blocked", "cmake needs a configured build tree; there is none here")

    Name the number, message or behaviour you read it from; "it worked" is not a
    reason. `fail` on a green run is expected sometimes and is not a setback — fix and
    re-run.

    `unknown` is the honest answer when the output does not settle the question, and it
    is a complete one: it stands as reported and counts against nothing. When the
    question is worth settling, it is also a starting point — say what would settle it
    and get it, in proportion to what the answer is worth — a reference implementation,
    a documented value, an analytical limit, an invariant, a refinement trend. Read the
    documentation, the literature or the repository, and fetch from the
    web if you can reach it. When the standard is the user's to set rather than yours to
    find — their conventions, their acceptance criteria, their data — ask them. Then
    report again. If it stays out of reach, say so and name what is unverified: you are
    not blocked on being certain, and the user is the last judge of what you could not
    settle.

    `blocked` is for a run that failed on a wall that is not in your change: a build to
    configure, a package that is not installed, a dataset, an allocation. It does not make
    the run a success — it stays reported as not completed, with your reason attached. It
    says only that repairing it is not the next piece of this work, which stops it counting
    against you as an unfinished task. Use it when reaching a first run would cost a step of
    its own; never to avoid a fix you could make.

    Args:
        verdict: "pass", "fail", "unknown" or "blocked".
        reason:  What in the output shows it — the number, message or behaviour.
        run:     Which run is being judged, as its command — a recognisable fragment is
                 enough. Optional: left out, a "pass" settles the most recent run, so
                 name the run whenever you are judging an earlier one. "fail" and
                 "unknown" address everything outstanding and never need it; "blocked"
                 addresses every failed run the same way.
    """
    value = (verdict or "").strip().lower()
    if value not in _VERDICT_VALUES:
        return err(f"verdict must be one of {', '.join(_VERDICT_VALUES)} — got '{verdict}'.")
    if not (reason or "").strip():
        return err("reason is required — name what in the output shows the verdict.")
    return ok({"verdict": value, "reason": reason.strip(), "run": (run or "").strip()})


if __name__ == "__main__":
    mcp.run()

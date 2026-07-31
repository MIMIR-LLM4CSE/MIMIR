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

THE 73 ALLOWED COMMANDS, BY CATEGORY
------------------------------------
One taxonomy, declared in ``_shared/shell_paths.py`` and read by both ends: the
allowlist below is the union of those groups, the client's classifier maps each to a
capability kind, and the category is what decides approval and plan-mode availability.
``bash_allowed_commands()`` reports it per command. The parity test in
``test_server_contracts.py`` fails if the two ends drift.

===========  =========  ========  ==================================================
category     in plan?   approval  commands
===========  =========  ========  ==================================================
neutral      yes        no        pwd echo which basename dirname realpath df
                                  true false :
read         yes        no        cat head tail nl sed◆ wc cut sort◆ uniq comm tr
                                  fold column cksum md5sum sha256sum stat file
search       yes        no        grep rg
inspect      yes        no        ls find du
chdir        yes        no        cd
env          query      mutation  module pip pip3 conda conda3 mamba mamba3  ◆
write        no         yes       mv cp mkdir
exec         no         yes       gcc g++ gfortran nvcc javac java node python
                                  python3 make cmake ctest pytest ruff mypy pyflakes
                                  black pdflatex latex xelatex lualatex pdftex tex
                                  bibtex biber makeindex latexmk dvips dvipdf
===========  =========  ========  ==================================================

◆ Per-call shifts: ``sed -i`` / ``sort -o FILE`` and a ``> file`` redirection turn a
side-effect-free command into a **write**; ``< file`` and fd redirection change nothing;
for the env managers the sub-command decides (query vs mutation), and an unknown one is
assumed to mutate. In a chain, one non-read-only segment makes the whole call
non-read-only. ``git`` (use the ``localgit`` server), deletion (``rm``/``rmdir``) and
every shell interpreter are excluded outright.

This server is intentionally restricted:
- only runs inside the workspace root
- only the allowlist above is accepted. It is the primary surface for
  compile/run/test/validate — invoke the toolchain directly with the exact flags the
  task needs. A rejection inlines the full allowlist in its payload: the reply to a
  shell call must never point at something that reads as another shell command
  to try.
- shell interpreters and generic runners (bash, sh, eval, env, xargs, sudo …)
  are permanently excluded — they nest an unvalidated command — and say so
  explicitly when rejected, so the agent stops looking for a wrapper
- command chaining is allowed (``;``, ``&&``, ``||``, ``|``) since real tasks
  often need several commands in one call; each command between separators is
  tokenized and validated against the allowlist independently
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
  expansion still work for the allowed command form
- the environment managers (``pip``, ``conda``, ``mamba``) are available, scoped to
  querying and adding: ``pip install``, ``conda create``/``conda env create`` and
  the local queries, but no ``uninstall``/``remove``/``clean`` (the write with no
  way back), no ``conda run`` (it nests an unvalidated command) and no
  ``config`` (it would persist settings past the session). An install writes into
  the target interpreter's site-packages — outside the workspace by construction,
  so the approval prompt, not a path check, is what governs it
- ``module`` (HPC Lmod) is supported: it is a shell *function*, so the server
  defines it by sourcing Lmod's init in the wrapper it builds around the
  validated command. This means ``module load cuda && nvcc ...`` works in a
  single call; a load does not persist to the next call (fresh subprocess)

Every call starts at the workspace root — there is no working-directory argument.
``cd`` is allowlisted and holds for the rest of a single call (``cd sub && pytest
t.py``), but each call is a fresh subprocess so it does not carry over. One way to
say "where", not two: a second mechanism was only ever a second thing to keep
confined, and the root is the one base every path in the call is judged against.

Critical hardening applied:
- one segmenter, shared with the client: ``shell_paths.parse_segments`` decides
  where a command ends and which tokens are a redirection rather than an argument,
  so the guard that *confines* a path and the gate that *prompts* for it can never
  read the same command two ways (see that module's docstring)
- treats an unquoted newline as the command separator it is, so every line of a
  multi-line command is validated against the allowlist (never folded into the
  argv of the line above)
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
from capabilities import tool_caps, PLAN_READONLY, CODE_EXEC, RECOVERABLE
from responses import err, ok
from shell_paths import (
    COMMAND_CATEGORIES as _COMMAND_CATEGORIES,
    ENV_MANAGER_COMMANDS as _ENV_MANAGER_COMMANDS,
    EXEC_COMMANDS as _EXEC_COMMANDS,
    INSPECT_COMMANDS as _INSPECT_COMMANDS,
    NEUTRAL_COMMANDS as _NEUTRAL_COMMANDS,
    READ_COMMANDS as _READ_COMMANDS,
    READONLY_NESTED_COMMANDS,
    SEARCH_COMMANDS as _SEARCH_COMMANDS,
    WRITE_COMMANDS as _WRITE_COMMANDS,
    ShellParseError as _ShellParseError,
    cd_destination as _cd_destination,
    expansion_operands,
    is_path_like_command as _is_path_like_command,
    normalize_path_arg as _normalize_path_arg,
    parse_segments as _parse_segments,
    segment_path_operands,
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
_DEFAULT_TIMEOUT = 10

# The user's real home, resolved once. Commands run with this as HOME so every tool
# reads the configuration the user actually has (see _safe_env). It is captured here
# because _safe_env rebuilds the environment from scratch, and because the validator
# needs the same value to know where a bare 'cd' would land.
_REAL_HOME = os.path.realpath(os.path.abspath(os.path.expanduser("~")))

# The allowlist is the union of the command *groups* in servers/_shared/shell_paths.py,
# which the client's gate and classifier import too: this guard *confines* the paths a
# command names, that gate *prompts* for the ones outside the workspace, and the
# classifier decides what is plan-safe. Two copies of the taxonomy would fail silently
# in both directions — a path the client misses cannot be granted, a path this guard
# misses is never gated, and a command one side knows the other refuses is dead weight
# advertised as a capability. Adding a command therefore means adding it to a *group*,
# and the parity test in test_server_contracts.py keeps the two ends honest.
_ALLOWED_COMMANDS = {
    # Compile / run / test / validate.
    *_EXEC_COMMANDS,
    # Read a file's content or metadata. 'sed' is dual-use: read-only when printing
    # (sed -n / sed 's///' to stdout) but a real *write* with in-place '-i'. Its paths
    # are confined either way, so an in-place edit cannot escape the workspace; the
    # client classifier flags 'sed -i' as a write, so it is approval-gated and rejected
    # in plan mode (never side-effect-free).
    *_READ_COMMANDS,
    # Search file contents — this server is the only grep surface there is.
    *_SEARCH_COMMANDS,
    # Enumerate the tree.
    *_INSPECT_COMMANDS,
    # File management. Real writes: they move, duplicate, create and re-mode inside
    # the workspace, and are approval-gated like any other write (rejected in plan
    # mode).
    # Every operand — source AND destination — is confined, so `cp /etc/passwd .` is
    # refused while reorganising the workspace's own files is not.
    # NOTE: deletion ('rm', 'rmdir') stays out — it is the one write with no recovery
    # path, and nothing in a build/benchmark flow needs it.
    *_WRITE_COMMANDS,
    # Metadata words and no-ops. The no-ops carry no side effect at all, but they are
    # what makes the standard capability probe expressible:
    # `which pdflatex 2>/dev/null || true` tests for a toolchain without the whole
    # chain being rejected on its last segment. Without them the agent has no way to
    # ask "is X available?" and ends up probing with commands that get refused one
    # after another.
    *_NEUTRAL_COMMANDS,
    # Where the rest of the call runs. One 'cd' rebases every later path in the chain.
    "cd",
    # HPC environment modules. 'module' is an Lmod *shell function*, not a
    # binary, so it cannot run under 'bash --norc' on its own — the server
    # defines it by sourcing Lmod's init in the (server-built, un-validated)
    # wrapper around the user command (see _module_preamble). This lets a single
    # call do 'module load cuda && nvcc ...': the load mutates the env of that
    # one shell, which the rest of the chain then inherits. Args are validated
    # against a fixed subcommand set below (discovery + load only).
    "module",
    # Environment managers (pip/conda/mamba). The agent can query and provision the
    # environment it runs in — install a missing package, create an env from a YAML —
    # scoped by _ENV_MANAGER_SUBCOMMANDS below in the same spirit as 'module': query
    # and add, never remove, and never a sub-command that nests another command
    # ('conda run', which would bypass this validator exactly as 'bash -c' would).
    #
    # These are the one group whose *effect* is deliberately outside the workspace:
    # an install writes into the target interpreter's site-packages, which no path
    # check here can (or should) confine. Two things bound it instead — the approval
    # prompt every call goes through, and the sub-command scope below. Note the
    # target: PATH puts the interpreter MIMIR runs under first (see _safe_env), so a
    # bare 'pip install' provisions *that* env unless the agent names another.
    *_ENV_MANAGER_COMMANDS,
}

# 'module' subcommands we accept: discovery (read-only) and load only. No
# destruction (unload/rm/swap/purge/reset) and no collection creation
# (save/restore) — a benchmark step should only inspect modules or bring one in,
# never tear down or persist the environment. Anything else is rejected.
_MODULE_SUBCOMMANDS = {
    # discovery / read-only
    "list", "avail", "av", "show", "display", "spider", "whatis",
    "help", "keyword", "overview", "is-loaded", "is-avail", "help",
    # load only
    "load", "add",
}

# Environment-manager sub-commands we accept, by family ('pip3' → pip, 'mamba' →
# conda). Query and add only, for the same reason 'module' gets no 'unload'/'purge':
# an uninstall or an env removal is the write with no recovery path, and nothing in a
# build/benchmark flow needs it. Also absent, and *not* for that reason:
# - 'conda run' / 'mamba run' — they nest an arbitrary command, so allowing them
#   would void this validator the way 'bash -c' would (see _SHELL_RUNNERS).
# - 'pip config' / 'conda config' — they persist settings outside the workspace and
#   outside the session, so a later call would run under an environment the user
#   never saw being changed.
# Anything unrecognised is rejected rather than passed through: a sub-command this
# server has not reasoned about is exactly the one that turns out to delete something.
_ENV_MANAGER_SUBCOMMANDS = {
    "pip": {"install", "download", "list", "show", "freeze", "check", "inspect",
            "index", "help"},
    "conda": {"install", "create", "update", "upgrade", "list", "info", "search",
              "env", "help"},
}

# 'conda env <sub>': the container sub-command needs its own gate, since it holds both
# a creation ('conda env create -f env.yml') and a removal.
_CONDA_ENV_SUBCOMMANDS = {"create", "list", "export", "update"}


def _env_manager_family(argv0: str) -> str:
    """Which sub-command table governs *argv0* ('pip3' → pip, 'mamba3' → conda)."""
    return "pip" if argv0.startswith("pip") else "conda"


def _validate_env_manager_args(argv: list[str]) -> dict | None:
    """Confine an environment manager to querying and adding (see the tables above)."""
    family = _env_manager_family(argv[0])
    allowed = _ENV_MANAGER_SUBCOMMANDS[family]
    positionals = [a for a in argv[1:] if not a.startswith("-")]
    sub = positionals[0] if positionals else None
    if sub is None:
        return err(
            f"'{argv[0]}' needs a sub-command.",
            hint=f"Accepted here: {', '.join(sorted(allowed))}.",
        )
    if sub not in allowed:
        return err(
            f"'{argv[0]} {sub}' is not allowed.",
            hint=f"This server scopes environment managers to querying and adding: "
                 f"{', '.join(sorted(allowed))}. Removal is not available (no "
                 f"uninstall/remove/clean — it is the write with no way back), and "
                 f"neither is a sub-command that runs another command or persists "
                 f"configuration. Report the limitation rather than looking for "
                 f"another spelling of it.",
        )
    if family == "conda" and sub == "env":
        env_sub = positionals[1] if len(positionals) > 1 else None
        if env_sub not in _CONDA_ENV_SUBCOMMANDS:
            named = f" {env_sub}" if env_sub else ""
            return err(
                f"'{argv[0]} env{named}' is not allowed.",
                hint=f"Accepted after 'env': {', '.join(sorted(_CONDA_ENV_SUBCOMMANDS))}.",
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
# allowlist to point at.
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
# (`find -exec`, `rg --pre`). Matched as an exact token or a glued '=' form (e.g.
# '--output=x'). Keyed by argv[0].
#
# NOTE: creating a file is not itself the reason a flag is here. A write flag whose
# target the guard can *see* is confined instead of denied — that is
# ``shell_paths.WRITE_VALUE_FLAGS_BY_CMD``, which covers ``gcc -o``, ``sort -o`` and
# friends in both the separated and glued spellings. What stays out is the write the
# guard cannot resolve at all: ``find -fprint`` (whose flag value the operand walk does
# not reach) and ``find -exec`` / ``rg --pre`` (a nested command, opaque by nature).
_FORBIDDEN_ARG_TOKENS_BY_CMD = {
    # -exec/-execdir are NOT here: the parser lifts their payload into its own
    # nested segment, which is then accepted or rejected on the nested command's own
    # head (see the segment.nested branch in _validate_command). -ok/-okdir stay —
    # they prompt on a tty the agent does not have. Everything else here hides its
    # target in flag-value position, where the operand walk never looks.
    "find": {"-ok", "-okdir", "-delete",
             "-fprint", "-fprint0", "-fprintf", "-fls", "--in-place"},
    "rg":   {"--pre", "--pre-glob", "--hostname-bin", "--search-zip", "-f", "--file"},
    # '-f/--file' reads the patterns from a file. Denylisted for both engines rather
    # than merely confined: it is a read whose operand sits in flag-value position,
    # which is exactly where the pattern-skipping rule stops looking.
    "grep": {"-f", "--file"},
    # Recursive mode changes. Every path chmod is given is confined to the workspace
    # like any other operand, so the concern is not *where* it reaches but how much it
    # rewrites from one token: '-R' re-modes a whole tree, which no build step needs
    # and which the user cannot review from the call. Single-path forms are the point
    # of allowing chmod at all (see shell_paths.WRITE_COMMANDS) and stay available.
    "chmod": {"-R", "--recursive"},
    **{cmd: _TEX_FORBIDDEN_FLAGS for cmd in _TEX_COMMANDS},
}

# Shell interpreters and generic command runners. Allowlisting any of them would
# void the whole validator (argv[0] is what gets checked), so they stay out — but
# they are the first thing a model reaches for, so they get a specific rejection
# telling it what to do instead, rather than the generic allowlist pointer.
_SHELL_RUNNERS = {
    "bash", "sh", "zsh", "ksh", "dash", "csh", "tcsh", "fish",
    "eval", "exec", "source", ".", "command", "env", "xargs", "nohup", "sudo",
}

# Commands whose exit status 1 means "nothing found" rather than "failed":
# `grep`/`rg` return 1 on no match, `which` returns 1 when the binary is absent.
# Both are legitimate *answers* to a question the agent asked. Reported as an
# error they read as a broken command, and the model re-runs them unchanged —
# so a clean no-match (rc 1, empty stdout) is reported as a successful result.
# A real failure (bad pattern, unreadable file) exits 2 and stays an error.
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
        # HOME is the user's real home, inherited. It used to be pinned to the
        # workspace, which looked like confinement but was not: what confines a
        # command here is the path check on the operands it names, and HOME changes
        # none of that. What pinning actually did was detach every tool from the
        # user's own configuration and caches, silently and per tool — user
        # site-packages (~/.local/lib) vanished from `python`, `pip` lost its config
        # and re-downloaded into a stray .cache/ in the repo, `conda` lost ~/.condarc
        # and its channels, and TeX stopped finding packages installed in ~/texmf.
        # The agent then saw failures the user cannot reproduce in their own shell.
        # Startup files are kept out by 'bash --noprofile --norc' plus BASH_ENV
        # below, which is what that concern actually needed; incidental cache writes
        # land where they would for the user, instead of polluting the workspace.
        "HOME": _REAL_HOME,
        # Prevent non-interactive bash from loading an arbitrary startup file.
        "BASH_ENV": "",
        "PWD": cwd,
        # kpathsea reads texmf.cnf settings from the environment, so these pin the
        # TeX engines' sandbox regardless of the site's texmf.cnf: no \write18 at
        # all, and 'paranoid' file access — TeX may only write below the cwd and
        # may not read dotfiles or absolute paths outside its trees. This is the
        # backstop for the shell-escape flags already denylisted at validation.
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
    """Validate a 'module ...' invocation against the allowed subcommand set."""
    if len(argv) < 2:
        return err(
            "'module' needs a subcommand.",
            hint="Use e.g. 'module avail', 'module list', or 'module load cuda'.",
        )
    sub = argv[1]
    if sub not in _MODULE_SUBCOMMANDS:
        return err(
            f"'module {sub}' is not a supported subcommand.",
            hint="Allowed: discovery (avail, list, show, spider, whatis) and "
                 "load/add only. Unload, swap, purge, save and restore are not "
                 "permitted.",
        )
    for arg in argv[2:]:
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

    ``cd`` is allowlisted, so it must be confined like any other path operand —
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
        # The *resolved* destination, not the token written: every refusal in this
        # server names the path the user would be asked to approve, so the message
        # and the prompt refer to the same thing ('..' and '/etc' do not).
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
    # Segmentation — which commands the shell would run, and which files their
    # redirections touch — comes from shell_paths.parse_segments, the module the
    # client's gate imports too (see the note on _ALLOWED_COMMANDS). What is left
    # here is the *policy*: the allowlist, the flag denylists, and confinement.
    try:
        segments = _parse_segments(command)
    except _ShellParseError as e:
        return err(e.message, hint=_PARSE_HINTS.get(e.reason, ""))

    parsed = []
    # The base for path confinement. A 'cd' segment moves it (within the
    # workspace), so later segments are checked against the directory the shell
    # will actually be in — see _resolve_cd_target.
    seg_cwd = cwd
    for segment in segments:
        argv = segment.argv
        # A program named by path is not judged by the allowlist — the allowlist is a
        # set of command *names*, and a path names a specific file instead. It is
        # judged as the path it is, by the confinement check below: inside the
        # workspace it runs, outside it needs the user's approval like any other
        # out-of-workspace path. What stays refused whatever the spelling is a shell
        # runner: allowing '/bin/sh' by path would reinstate the very bypass the
        # name-based refusal exists to prevent, and approving one root would unlock it.
        argv0_is_path = _is_path_like_command(argv[0])
        runner = os.path.basename(argv[0]) if argv0_is_path else argv[0]
        if argv[0] not in _ALLOWED_COMMANDS and (
            not argv0_is_path or runner in _SHELL_RUNNERS
        ):
            if runner in _SHELL_RUNNERS:
                hint = (
                    f"'{argv[0]}' runs an arbitrary nested command, which would "
                    "bypass this validator, so it can never be allowed. Write the "
                    "command out directly instead — chaining (';', '&&', '||', "
                    "'|') is supported, so anything you would put in a script can "
                    "be expressed as one chain of approved commands. To run a "
                    "script that already exists in the workspace, invoke it by "
                    "path ('./build.sh'), adding 'chmod +x ./build.sh' first if it "
                    "is not executable."
                )
            else:
                hint = (
                    "This command is not part of the toolchain available here, and "
                    "no wrapper will make it available — do not retry a variant of "
                    "it. Either use one of the approved commands below, or report "
                    "that the capability is unavailable. To run a binary you "
                    "compiled, reference it by a workspace path such as './a.out'."
                )
            # The approved list is inlined rather than pointed at: this payload is
            # the reply to a *shell* call, so any 'call X() to find out' pointer
            # reads as another shell command to try — which is how the agent ends
            # up running the name of an MCP tool and failing again on it.
            return err(
                f"Command '{argv[0]}' is not allowed.",
                hint=hint,
                allowed_commands=sorted(_ALLOWED_COMMANDS),
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
        # command: allowed only if its head is read-only. The flag denylist below
        # would otherwise reject it on the `-exec` token alone, without ever looking
        # at what is nested — which dead-ended read-only fan-out, since the hint's
        # suggested workaround needs `xargs` and that is permanently banned.
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
            hint="Use a narrower search scope or a shorter command.",
            cwd=cwd,
        )
    except Exception as e:
        return err(str(e), cwd=cwd)


@mcp.tool()
def bash_allowed_commands() -> dict:
    """Return the allowlist of commands supported by this server, with their category.

    Use this before bash_run when you are unsure whether a command is permitted.

    The category says what running it does, which is also what decides how it is
    gated: 'read', 'search', 'inspect', 'neutral' and 'chdir' are side-effect-free —
    they run unattended and are available while planning. 'write', 'exec' and 'env'
    (pip/conda/module) prompt for approval and are unavailable while planning. Two
    things shift a command out of its category for one call: an in-place/output flag
    ('sed -i', 'sort -o') and a redirection to a file ('ls > out.txt') make it a
    write, while an 'env' command's sub-command decides between query and mutation.
    """
    return ok({
        "commands": sorted(_ALLOWED_COMMANDS),
        "categories": {c: _COMMAND_CATEGORIES.get(c, "exec")
                       for c in sorted(_ALLOWED_COMMANDS)},
        "count": len(_ALLOWED_COMMANDS),
    })


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

        grep -rn "FastMCP" mimir            # search; rg works too
        sed -n '1,40p' path/to/file.py      # read a line range
        gcc solver.c -O3 -o solver.out -lm && ./solver.out
        python -m pytest tests/test_thing.py -q     # also ruff / mypy / py_compile
        ./solver.out > run.log 2>&1 && tail -20 run.log
        module load cuda && nvcc kernel.cu -o kernel.out
        pip list | grep -i numpy            # then: pip install <pkg>

    Gating: read-only commands (search, read, inspect, `pip list`, `module avail`) run
    unattended and are available while planning. Writes, execution and installs prompt
    for approval and are blocked while planning. `bash_allowed_commands()` reports the
    category of every command; a rejection inlines the whole allowlist, so there is
    never a reason to go hunting for it.

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
    - no shell interpreter or wrapper (bash, sh, eval, env, xargs, sudo): write the
      command out directly, chaining covers what a script would do. No heredoc — pass
      the text as a quoted argument. No backgrounding, substitution or subshell.
    - no deletion ('rm'), no uninstall/remove for pip/conda, no git (use the
      'localgit' server). These are absent by design: report the limitation instead of
      trying another spelling.
    - 'sed -i' rewrites in place (prefer the file-edit tools); a bare 'pip install'
      provisions the interpreter MIMIR itself runs under.
    - an empty result from a probe ('which pdflatex') is a conclusive "absent", not an
      error to retry.

    Args:
        command: Shell command string (single command or simple pipeline).
        timeout: Timeout in seconds.
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
    timeout = max(1, min(timeout, 30))
    result = _run(command, cwd, timeout, preamble=preamble, segments=validation["segments"])
    if timeout != requested_timeout:
        result["timeout_clamped"] = True
        result["requested_timeout"] = requested_timeout
    return result


if __name__ == "__main__":
    mcp.run()

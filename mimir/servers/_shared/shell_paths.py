"""How a shell command splits into commands, and which paths its arguments name.

Shared by the bash server, which confines those paths, and the client, which prompts
for the outside ones: they must agree. Stdlib only. Whether a path is allowed is the
caller's decision.
"""

import functools
import os
import re
import shlex
import shutil
from dataclasses import dataclass, field
from collections.abc import Mapping

# The command taxonomy, split by *effect* — read, search, inspect, write, execute,
# provision. It is a *classification*, not a gate: bash runs anything outside
# DENIED_COMMANDS, and these groups say what a command's effect is, which decides
# approval (read-only is waived) and plan-mode availability. A head in no group is
# still runnable — it classifies as unknown, hence always approval-gated.

# Build / execution. An interpreter or compiler takes a file operand exactly as ``cat``
# does, so ``python /tmp/evil.py`` is confined for the same reason ``cat /etc/passwd``
# is — and it additionally *executes* what it reads. The sandbox treats all four groups
# below identically; they are kept apart because what a green exit *proves* differs, and
# that difference is what the client's evidence layer keys off (see EXEC_EFFECTS).

# Checkers: they read the code, emit diagnostics, and exit non-zero when they found
# something. Nothing of the project runs, so there is no output for anyone to judge —
# the exit code is the whole finding. A compiler belongs here even when it emits an
# object: what it *reports* is diagnostics.
VALIDATOR_COMMANDS = frozenset({
    "gcc", "g++", "gfortran", "nvcc", "javac",
    "clang", "clang++", "icc", "icpc", "ifort", "ifx",
    # MPI compiler wrappers: they drive one of the compilers above. Named here so the
    # platform probe cannot advertise a toolchain whose runs owe no verdict.
    "mpicc", "mpicxx", "mpic++", "mpiCC", "mpifort", "mpif90", "mpif77",
    # Lint / typecheck / format, invocable by name or via ``python -m <tool>``. These
    # are the validator surface — there are no dedicated code-quality tools.
    "ruff", "pyflakes", "mypy", "black",
})

# Build drivers: they orchestrate compilers over a whole project. A Makefile recipe is
# effectively arbitrary execution, but no more so than the gcc already above, and like a
# compiler it is judged by its exit code — a build either produced its artefacts or said
# why not. TeX is a build for the same reason: it compiles a document.
BUILD_COMMANDS = frozenset({
    "make", "cmake", "pmake", "ninja", "meson",
    "pdflatex", "latex", "xelatex", "lualatex", "pdftex", "tex",
    "bibtex", "biber", "makeindex", "latexmk", "dvips", "dvipdf",
})

# Executions proper: they run the project's code, so exit 0 means the program reached
# its end and nothing more. What it printed is the result, and only a reader can say
# whether that result is right. A test runner is here, not above: a suite is a run
# judged on its own output, however green it exits.
RUN_COMMANDS = frozenset({
    "python", "python3", "java", "node", "pytest", "ctest",
})

# ``source env.sh`` runs the file's contents in the *current* shell, so the rest of the
# chain sees the environment it sets — which is why './env.sh' cannot replace it. It
# executes what it reads, like every entry above, and its file operand is confined the
# same way. The ``.`` spelling stays refused (see SHELL_INTERPRETERS). What it proves about
# the project, though, is nothing: it is preparation, not evidence.
ENV_SETUP_COMMANDS = frozenset({"source"})

EXEC_COMMANDS = (
    VALIDATOR_COMMANDS | BUILD_COMMANDS | RUN_COMMANDS | ENV_SETUP_COMMANDS
)

# head → which of the four groups it belongs to, for the client's evidence layer. Kept
# here beside the groups so a command added to one of them cannot be forgotten in the
# mapping; the sandbox side keeps consulting ``EXEC_COMMANDS`` alone.
EFFECT_VALIDATE: str = "validate"
EFFECT_BUILD: str = "build"
EFFECT_RUN: str = "run"
EFFECT_ENV_SETUP: str = "env_setup"

EXEC_EFFECTS = {
    **{c: EFFECT_VALIDATE for c in VALIDATOR_COMMANDS},
    **{c: EFFECT_BUILD for c in BUILD_COMMANDS},
    **{c: EFFECT_RUN for c in RUN_COMMANDS},
    **{c: EFFECT_ENV_SETUP for c in ENV_SETUP_COMMANDS},
    # Reached only as ``python -m py_compile``: a checker with no command of its own,
    # so it is deliberately absent from EXEC_COMMANDS (nothing should file it as a bare
    # head) while still being named here, where effects are read.
    "py_compile": EFFECT_VALIDATE,
}

# Package/env provisioning. Kept apart from EXEC_COMMANDS ("compiles or runs the
# project's code") because these mutate the *interpreter environment* instead. They
# still take file operands (requirements file, local wheel, env YAML, ``pip install .``)
# so they are path-sensitive; what they install lands in site-packages, outside the
# workspace by construction, governed by the approval prompt rather than this module.
ENV_MANAGER_COMMANDS = frozenset({
    "conda", "conda3", "mamba", "mamba3", "pip", "pip3",
})

# ``sed`` and ``sort`` sit here, not in WRITE: they only write when carrying an
# in-place/output flag, which the client detects per call (a group cannot express
# "depends on a flag").
READ_COMMANDS = frozenset({
    "cat", "head", "tail", "nl", "sed", "wc", "cut", "sort", "uniq", "comm",
    "tr", "fold", "column", "cksum", "md5sum", "sha256sum",
    # These open the file they are given, unlike the no-ops below whose args are words.
    "stat", "file",
})
SEARCH_COMMANDS = frozenset({"grep", "rg"})
INSPECT_COMMANDS = frozenset({"ls", "find", "du"})
# ``git`` is placed by its subcommand: these only read the repository (anything that
# writes, moves a ref, touches the worktree or a remote, and ``config``, is absent).
GIT_READ_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "branch", "tag", "remote",
    "rev-parse", "rev-list", "ls-files", "ls-tree", "cat-file",
    "blame", "describe", "shortlog", "whatchanged", "reflog", "grep",
})
# git's own flags that consume the token after them; skipped when looking for the
# subcommand so ``git -C /repo status`` is still a status.
GIT_VALUE_FLAGS = frozenset({"-C", "--git-dir", "--work-tree", "--namespace", "-c"})
# Unconditional writes. ``chmod`` writes a file's *mode* rather than its contents and is
# here for one reason: a script arriving without the x bit (fresh checkout, or one the
# agent just wrote) otherwise dead-ends on "Permission denied" with no allowed command
# that fixes it. It grants no new *kind* of power — ``python f.py``, ``make`` and a
# compiled ``./a.out`` already execute workspace code.
WRITE_COMMANDS = frozenset({"mv", "cp", "mkdir", "chmod"})
# No side effect, and no discovery evidence either. The no-ops are what makes a
# capability probe expressible end to end (``which pdflatex || true``).
NEUTRAL_COMMANDS = frozenset({
    "pwd", "echo", "which", "basename", "dirname", "realpath", "df",
    "true", "false", ":",
    # Prints its format and arguments, like ``echo``: they are text, and a script
    # passed line by line would otherwise have every quoted line judged as a path.
    "printf",
    # Read and set the environment. Their arguments are words (``VAR=value``), not
    # paths, and unlike ``env`` neither can run a command; an ``export`` holds only
    # for the rest of the one chain, which is fresh per call.
    "printenv", "export",
})

# ── What bash refuses outright ─────────────────────────────────────────────────
#
# Everything not named below runs, subject to the user's approval prompt and to path
# confinement. These three sets are the exceptions, each for its own structural reason,
# and each rejection names the route that replaces it.

# They execute a command this validator never sees, which would void segmentation —
# and with it path confinement and every classification keyed on the head. Refused in
# any spelling, including by absolute path and behind a wrapper. ``source`` is the
# deliberate exception (ENV_SETUP_COMMANDS): its operand is a *file*, confined like any
# other path. Only the readable spelling is offered, so ``.`` stays here.
SHELL_INTERPRETERS = frozenset({
    "bash", "sh", "zsh", "ksh", "dash", "csh", "tcsh", "fish",
    "eval", "exec", ".", "command", "sudo", "su", "doas",
})

# Destroy data outside any notion of a workspace, or repartition the machine. No
# approval prompt makes these reviewable from a command line, and no coding task needs
# them. (``rm``/``rmdir`` are deliberately NOT here: destructive but reviewable, so they
# run under the approval prompt with their operands confined.)
DESTRUCTIVE_COMMANDS = frozenset({
    "shred", "dd", "mkfs", "fdisk", "parted", "swapoff", "mkswap",
})

# Job submission has one route: the typed HPC tools, which return the background-job
# descriptor the client's watcher polls. A second route through bash would submit jobs
# nothing tracks. Cancellation goes the same way, to a tool that takes one job the user
# owns: `scancel -u me` or `--partition` would cancel far more than one card shows.
CLUSTER_SUBMIT_COMMANDS = frozenset({"sbatch", "salloc", "scancel"})

DENIED_COMMANDS = (
    SHELL_INTERPRETERS | DESTRUCTIVE_COMMANDS | CLUSTER_SUBMIT_COMMANDS
)


# ── Wrappers: commands whose argument list is another command ──────────────────
#
# ``timeout 60 pytest -q`` is a pytest run, and must be validated and classified as
# one. Unwrapping is what lets these through without letting them smuggle anything: the
# inner head is re-tested against DENIED_COMMANDS at every layer, so ``timeout 5 bash
# -c x`` is refused on ``bash``.

# Wrapper flags that take a separate value, which must be skipped with the flag or the
# value would be mistaken for the inner command. Glued (``-n1``) and ``=`` forms need no
# entry — they are one token.
_WRAPPER_VALUE_FLAGS = {
    "timeout": ("-k", "--kill-after", "-s", "--signal"),
    "nice": ("-n", "--adjustment"),
    "ionice": ("-c", "--class", "-n", "--classdata", "-p", "--pid"),
    "stdbuf": ("-i", "--input", "-o", "--output", "-e", "--error"),
    "xargs": ("-I", "-i", "--replace", "-n", "--max-args", "-L", "-P", "--max-procs",
              "-d", "--delimiter", "-E", "-s", "--max-chars", "-a", "--arg-file"),
    "nohup": (),
    "env": (),
    "srun": ("-n", "--ntasks", "-N", "--nodes", "-c", "--cpus-per-task",
             "-p", "--partition", "-t", "--time", "-J", "--job-name",
             "-A", "--account", "-w", "--nodelist", "--gres", "--mem",
             "--ntasks-per-node", "--cpu-bind", "-o", "--output", "-e", "--error"),
    "mpirun": ("-n", "-np", "--n", "--np", "-c", "-N", "--host", "--hostfile",
               "--machinefile", "--map-by", "--bind-to", "-x"),
    "mpiexec": ("-n", "-np", "--n", "--np", "-c", "-N", "--host", "--hostfile",
                "--machinefile", "--map-by", "--bind-to", "-x"),
}
COMMAND_WRAPPERS = frozenset(_WRAPPER_VALUE_FLAGS)

# Wrappers whose first bare positional is their own argument, not the command:
# ``timeout 5s cmd``. Recognised by shape (a duration), so ``timeout cmd`` — invalid
# anyway — does not silently eat the command.
_WRAPPER_LEADING_DURATION = frozenset({"timeout"})
_DURATION_RE = re.compile(r"^\d+(?:\.\d+)?[smhd]?$")

# A wrapper chain deeper than this is not a shape any real call has; the bound is what
# keeps unwrapping terminating on adversarial input.
_MAX_UNWRAP_DEPTH = 4


def _unwrap_once(argv: list[str]) -> list[str] | None:
    """The command *argv* wraps, or None when argv[0] is not a wrapper (or wraps nothing)."""
    if not argv:
        return None
    head = os.path.basename(argv[0]) if is_path_like_command(argv[0]) else argv[0]
    if head not in COMMAND_WRAPPERS:
        return None
    value_flags = _WRAPPER_VALUE_FLAGS[head]
    took_duration = False
    i, n = 1, len(argv)
    while i < n:
        tok = argv[i]
        if tok.startswith("-") and tok != "-":
            i += 2 if tok in value_flags and i + 1 < n else 1
            continue
        # 'env A=B cmd': assignments precede the command.
        if head == "env" and "=" in tok and not tok.startswith("="):
            i += 1
            continue
        if head in _WRAPPER_LEADING_DURATION and not took_duration and _DURATION_RE.match(tok):
            took_duration = True
            i += 1
            continue
        break
    inner = argv[i:]
    # No inner command: the wrapper is doing its own job ('env' prints the environment).
    return inner or None


def unwrap_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """(*argv* with its wrappers stripped, the wrapper heads that were stripped).

    Callers validate and classify the returned argv as if the wrappers were not there,
    which is what the shell effectively does — while the wrapper list lets a caller that
    cares (a rejection message) name what it went through.
    """
    wrappers: list[str] = []
    current = list(argv)
    for _ in range(_MAX_UNWRAP_DEPTH):
        inner = _unwrap_once(current)
        if inner is None:
            break
        wrappers.append(current[0])
        current = inner
    return current, wrappers


# Wrappers that request cluster resources of their own when nothing already holds an
# allocation: outside one, ``srun`` is a submission — it queues, then spends node hours —
# and must be treated like the typed submitters. Inside one (``SLURM_JOB_ID`` set, as
# under salloc or in a batch job) it launches a step on resources already granted.
ALLOCATING_WRAPPERS = frozenset({"srun"})
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_]\w*=")
_ALLOCATING_TEXT_RE = re.compile(r"(?:^|[\s;&|(])(?:\S*/)?srun(?:\s|$)")


def allocates_cluster(command: str, env: Mapping[str, str] | None = None) -> bool:
    """Whether running *command* here would request a new cluster allocation.

    A command the parser cannot read is scanned textually instead — erring toward
    "allocates", since what it guards is the most expensive action there is.
    """
    env = os.environ if env is None else env
    if env.get("SLURM_JOB_ID") or not isinstance(command, str):
        return False
    try:
        segments = parse_segments(command)
    except ShellParseError:
        return bool(_ALLOCATING_TEXT_RE.search(command))
    for seg in segments:
        argv = list(seg.argv)
        while argv and _ASSIGNMENT_RE.match(argv[0]):
            argv = argv[1:]
        if not argv:
            continue
        inner, wrappers = unwrap_argv(argv)
        heads = [os.path.basename(h) for h in [*wrappers, *inner[:1]]]
        if any(h in ALLOCATING_WRAPPERS for h in heads):
            return True
    return False


def denied_command(argv0: str) -> str | None:
    """The denylisted name *argv0* resolves to, or None.

    Matches by basename too, so ``/bin/bash`` is the ``bash`` that is refused.
    """
    if not argv0:
        return None
    for name in (argv0, os.path.basename(argv0)):
        if name in DENIED_COMMANDS:
            return name
    return None


# What may appear as a nested command (``find … -exec CMD {} \;``). Writers, execs and
# provisioners stay out: a nested command's operands include the ``{}`` placeholder,
# which no caller can resolve, so a nested write goes somewhere unknowable. Reading one
# is harmless — worst case, some file inside the allowed roots is read.
READONLY_NESTED_COMMANDS = frozenset(
    READ_COMMANDS | SEARCH_COMMANDS | INSPECT_COMMANDS | NEUTRAL_COMMANDS
)

# The commands whose arguments are *not* paths: their operands are words (``echo
# $HOME``), or character sets (``tr 'a-z' 'A-Z'``). Listing the exceptions rather than
# the path-takers is what makes an unknown command's operands confined by default —
# with a denylist, an unlisted head is the common case, and a head nobody classified is
# exactly the one whose operands most need the confinement. ``realpath`` is a no-op
# whose argument *is* a path, hence its exclusion from the exception.
PATH_INSENSITIVE_COMMANDS = frozenset((NEUTRAL_COMMANDS - {"realpath"}) | {"tr"})

# Every head the taxonomy above places. Used where a rule needs to know whether the
# operand shapes are *understood* rather than merely present — see expansion_operands.
KNOWN_COMMANDS = frozenset(
    READ_COMMANDS | SEARCH_COMMANDS | INSPECT_COMMANDS | WRITE_COMMANDS
    | EXEC_COMMANDS | ENV_MANAGER_COMMANDS | NEUTRAL_COMMANDS | {"cd", "module"}
)


_TRAILING_SEPARATOR = re.compile(r"(?:^|\s)(?:&&|\|\||\||;|&)$")
_LEADING_SEPARATOR = re.compile(r"^(?:&&|\|\||\||;)(?:\s|$)")


def flatten_unquoted_newlines(command: str) -> str:
    """*command* with each unquoted newline turned into ``;``, for validation only.

    So every line is judged as its own command; quoted newlines stay data and a
    backslash-newline is removed, as the shell does.
    """
    parts: list[str] = []
    buf: list[str] = []
    in_single = in_double = False
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_single and i + 1 < n:
            nxt = command[i + 1]
            if nxt in "\r\n":  # line continuation: the shell removes both
                i += 2
                if nxt == "\r" and i < n and command[i] == "\n":
                    i += 1
                continue
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "\r\n" and not in_single and not in_double:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))

    # Each part is quote-balanced (a boundary only lands outside quotes), so the
    # whitespace around a boundary is unquoted and safe to strip; blank lines drop.
    kept = [p.strip() for p in parts]
    kept = [p for p in kept if p]
    if not kept:
        return command.strip()
    out = kept[0]
    for part in kept[1:]:
        joined = bool(_TRAILING_SEPARATOR.search(out) or _LEADING_SEPARATOR.match(part))
        out = f"{out} {part}" if joined else f"{out} ; {part}"
    return out


def unquoted_substitution_marker(command: str) -> str | None:
    """The first substitution marker the shell would act on, read on the raw string, or None.

    Single quotes make markers text; double quotes do not, except for ``<(``/``>(``.
    """
    i, n = 0, len(command)
    in_single = in_double = False
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_single and i + 1 < n:
            i += 2  # the shell drops the backslash and takes the next char literally
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if not in_single:
            for marker in SUBSTITUTION_MARKERS:
                if in_double and marker in _PROCESS_SUBSTITUTION:
                    continue
                if command.startswith(marker, i):
                    return marker
        i += 1
    return None


# Characters bash still acts on inside double quotes. Everything else in a double-quoted
# string is text, parentheses and redirection arrows included.
_DOUBLE_QUOTE_ACTIVE = frozenset("$`")


def unquoted_shell_char(command: str, chars: frozenset[str]) -> str | None:
    """The first of *chars* the shell would act on in *command* (quote- and escape-aware), or None."""
    i, n = 0, len(command)
    in_single = in_double = False
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_single and i + 1 < n:
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            i += 1
            continue
        if ch in chars and not in_single and not (
            in_double and ch not in _DOUBLE_QUOTE_ACTIVE
        ):
            return ch
        i += 1
    return None


# ── Segmentation: one shell string → the commands the shell would run ──────────

# Separators that join several commands in one call. Each command between them is
# a separate argv, validated (server) and classified (client) on its own.
# Whether any of these commands exists on this machine. Cached: the answer cannot change
# usefully within a call, and both sides ask it per segment.
@functools.lru_cache(maxsize=None)
def any_command_on_path(commands: tuple[str, ...]) -> bool:
    return any(shutil.which(c) for c in commands)


COMMAND_SEPARATORS = frozenset({";", "&&", "||", "|"})

# Markers that run or inline *another* command — refused outright, since what the
# head tells us nothing about what the substitution executes.
# Only where the shell would *act* on them: inside single quotes these are ordinary
# text (a search pattern is the usual place). See :func:`unquoted_substitution_marker`.
SUBSTITUTION_MARKERS = ("$(", "`", "${", "<(", ">(")
# The two markers bash reads only outside any quotes.
_PROCESS_SUBSTITUTION = frozenset({"<(", ">("})

# Characters that make a token a bare shell operator rather than an argument.
_OPERATOR_CHARS = "();<>|&"

# Redirection targets that name no file: a file-descriptor reference ('1', '2-',
# '-') or the null sink. These only silence or merge a stream, so they carry no
# path and no side effect.
_FD_SINKS = ("/dev/null",)


class ShellParseError(Exception):
    """Raised by :func:`parse_segments` for a command it will not segment.

    Carries a stable *reason* slug so each caller can react in its own idiom — the
    server turns it into a rejection payload with a hint, the client treats it as
    "opaque" — without either re-deriving *why* the parse failed.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass(frozen=True)
class Redirection:
    """One redirection attached to a command: ``> out.txt``, ``< in.txt``, ``>> log``.

    Only redirections naming a *file* are recorded; fd dups and ``/dev/null`` are
    dropped during parsing since they name no path. *writes* distinguishes the
    direction, which is what decides whether the segment gained a side effect.
    """
    op: str
    target: str
    writes: bool


@dataclass(frozen=True)
class ParsedSegment:
    """One command of a chain: its argv and the files its redirections touch.

    ``nested`` marks one lifted out of another's arguments (``find -exec``); ``sep`` is the
    separator after it ("" for the last).
    """
    argv: list[str]
    redirections: list[Redirection] = field(default_factory=list)
    nested: bool = False
    sep: str = ""

    @property
    def write_targets(self) -> list[str]:
        return [r.target for r in self.redirections if r.writes]

    @property
    def read_targets(self) -> list[str]:
        return [r.target for r in self.redirections if not r.writes]

    @property
    def redirect_targets(self) -> list[str]:
        return [r.target for r in self.redirections]


# Flags whose value is a whole nested command rather than an argument.
_NESTED_COMMAND_FLAGS = frozenset({"-exec", "-execdir"})
# What ends a `find -exec` payload. `;` is otherwise a chaining separator, so it must
# be consumed here or the command splits in the wrong place.
_EXEC_TERMINATORS = frozenset({";", "+"})


def _tokenize(command: str) -> list[str]:
    """Quote-aware tokens of *command*; refuses substitution, which only the quoted text
    can show (shlex strips the quotes)."""
    if not isinstance(command, str) or not command.strip():
        raise ShellParseError("empty", "Command must be a non-empty string.")
    command = flatten_unquoted_newlines(command)
    if unquoted_substitution_marker(command) is not None:
        raise ShellParseError(
            "substitution", "Command/parameter substitution is not allowed.")
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""  # '#' is a literal char, not a comment start
        tokens = list(lexer)
    except ValueError as e:
        raise ShellParseError("quoting", f"Invalid shell quoting: {e}") from e
    if not tokens:
        raise ShellParseError("empty", "Invalid command syntax.")
    return tokens


def _is_operator(tok: str) -> bool:
    return bool(tok) and all(ch in _OPERATOR_CHARS for ch in tok)


def _operator_error(tok: str) -> ShellParseError:
    # An unquoted parenthesis starts a subshell whatever was meant; for a `find`
    # expression, saying which shape works saves a planning phase a detour.
    hint = ""
    if tok in ("(", ")"):
        hint = (" An unquoted parenthesis starts a subshell. For a `find`"
                " expression, drop the grouping: `-name a -o -name b` already"
                " means either.")
    return ShellParseError("operator", f"Shell operator '{tok}' is not allowed.{hint}")


class _SegmentBuilder:
    """Folds tokens into segments: argv, redirections, separators, and `find -exec`
    payloads, which become nested segments after the chain."""

    def __init__(self) -> None:
        self.segments: list[ParsedSegment] = []
        self.argv: list[str] = []
        self.redirections: list[Redirection] = []
        # Open while inside a `-exec` payload: its tokens belong to a nested command, and
        # the `;`/`+` ending it is a terminator, not a separator.
        self.nested_argv: list[str] | None = None
        self.nested: list[ParsedSegment] = []

    def feed(self, tokens: list[str], i: int) -> int:
        """Consume the token at *i* (and any it owns); return the next index."""
        tok = tokens[i]
        if self.nested_argv is not None and not self._redirects(tok):
            return self._nested_token(tok, i)
        if tok in _NESTED_COMMAND_FLAGS and self.argv:
            # Dropped from find's argv: keeping it would trip the flag denylist on the
            # very token whose payload has just been made inspectable.
            self.nested_argv = []
            return i + 1
        if tok in COMMAND_SEPARATORS:
            if not self.argv:
                raise ShellParseError(
                    "separator", "A command separator must join two real commands.")
            self.segments.append(ParsedSegment(self.argv, self.redirections, sep=tok))
            self.argv, self.redirections = [], []
            return i + 1
        if self._redirects(tok):
            return self._redirection(tokens, i)
        if _is_operator(tok):
            raise _operator_error(tok)
        self.argv.append(tok)
        return i + 1

    @staticmethod
    def _redirects(tok: str) -> bool:
        return _is_operator(tok) and ("<" in tok or ">" in tok)

    def _nested_token(self, tok: str, i: int) -> int:
        assert self.nested_argv is not None
        if tok in _EXEC_TERMINATORS:
            if self.nested_argv:
                self.nested.append(ParsedSegment(self.nested_argv, [], nested=True))
            self.nested_argv = None
        elif _is_operator(tok):
            # The shell reads it as its own and ends find's command first.
            raise ShellParseError(
                "separator", f"Shell operator '{tok}' inside -exec/-execdir ends the "
                "command before its ';' or '+'.")
        else:
            self.nested_argv.append(tok)
        return i + 1

    def _redirection(self, tokens: list[str], i: int) -> int:
        """A redirection belongs to the outer command, inside a payload too (the shell
        applies it to find). An fd dup or the null sink names no file."""
        tok = tokens[i]
        if tok.startswith("<<"):
            # A heredoc body is not an argv; its lines would each read as a command.
            raise ShellParseError(
                "heredoc", f"Redirection operator '{tok}' is not supported.")
        # The source fd ('2' in '2>/dev/null') was tokenized on its own.
        words = self.nested_argv if self.nested_argv is not None else self.argv
        if words and len(words[-1]) == 1 and words[-1].isdigit():
            words.pop()
        target = tokens[i + 1] if i + 1 < len(tokens) else None
        if target is None or target in COMMAND_SEPARATORS or _is_operator(target):
            raise ShellParseError("redirect", "Redirection is missing a target.")
        if not (target == "-" or target.rstrip("-").isdigit() or target in _FD_SINKS):
            self.redirections.append(Redirection(tok, target, writes=">" in tok))
        return i + 2

    def finish(self) -> list[ParsedSegment]:
        if self.nested_argv is not None:
            raise ShellParseError(
                "separator",
                "A nested command (-exec/-execdir) must be terminated by ';' or '+'.")
        if self.argv:
            self.segments.append(ParsedSegment(self.argv, self.redirections))
        elif self.redirections:
            raise ShellParseError("redirect", "A redirection must belong to a command.")
        if not self.segments:
            raise ShellParseError("empty", "Invalid command syntax.")
        # Nested commands trail the chain, so top-level order stays intact for `cd`.
        return self.segments + self.nested


def parse_segments(command: str) -> list[ParsedSegment]:
    """Split *command* into the commands the shell would run, with their redirections.

    Quote-aware; redirections are lifted out of argv (fd dups and ``/dev/null`` vanish).
    Raises :class:`ShellParseError` on substitution, subshells, backgrounding, heredocs or
    dangling operators.
    """
    tokens = _tokenize(command)
    builder = _SegmentBuilder()
    i = 0
    while i < len(tokens):
        i = builder.feed(tokens, i)
    return builder.finish()


def unquoted_pipe_offsets(command: str) -> list[int]:
    """Offsets of each unquoted ``|`` in *command* (not ``||``), to splice at the k-th pipe."""
    offsets: list[int] = []
    in_single = in_double = False
    i, n = 0, len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_single and i + 1 < n:
            i += 2  # the escape and whatever it protects, '|' included
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "|" and not in_single and not in_double:
            if i + 1 < n and command[i + 1] == "|":
                i += 2  # '||' chains two commands, it pipes nothing
                continue
            offsets.append(i)
        i += 1
    return offsets


def is_path_like_command(argv0: str) -> bool:
    """True if *argv0* references a program by path (``./a.out``) rather than name.

    Existence is intentionally not checked: validation runs before a chained
    compile step has produced the binary. Such a command executes whatever it
    points at, so its own operands are path-sensitive too.
    """
    return bool(argv0) and (os.sep in argv0 or argv0.startswith("." + os.sep))


def normalize_path_arg(raw_arg: str, cwd: str) -> str | None:
    """The real path a path-like argument names, against *cwd*; None for options, ``-`` and URLs."""
    arg = (raw_arg or "").strip()
    if not arg or arg == "-":
        return None
    if arg.startswith("-"):
        return None
    if "://" in arg:
        return None

    expanded = os.path.expanduser(os.path.expandvars(arg))
    if os.path.isabs(expanded):
        return os.path.realpath(expanded)
    return os.path.realpath(os.path.abspath(os.path.join(cwd, expanded)))


# Flags whose *value* is a pattern or a script — text, never a path to open.
# Skipping their value is what stops `grep -e /etc/passwd f.txt` from being read as
# a request for /etc/passwd when it is really a string to search for.
_TEXT_VALUE_FLAGS = {
    "grep": ("-e", "--regexp"),
    "rg": ("-e", "--regexp"),
    "sed": ("-e", "--expression"),
}

# Flags whose value IS a file the command opens (a pattern file, a sed script).
# They must stay path operands, or `rg -f /etc/patterns` would read out of bounds.
_FILE_VALUE_FLAGS = ("-f", "--file")

# Commands whose first bare positional is a pattern/script rather than a file:
# `grep foo bar.py`, `sed 's/a/b/' f.py`. Dropping it is what keeps a search for
# the *string* "/etc/passwd" from being mistaken for a read of that file. Only the
# first is dropped, and only when no -e/-f supplied the pattern instead.
_LEADING_PATTERN_COMMANDS = ("grep", "rg", "sed")

# Flags whose value is a file the command writes, so the glued form (``-o/tmp/x``,
# ``--output-file=...``) is confined like the separate one. Read paths (``-I``, ``-L``)
# stay unconfined on purpose; add only write flags here.
WRITE_VALUE_FLAGS_BY_CMD = {
    "gcc": ("-o",), "g++": ("-o",), "gfortran": ("-o",), "nvcc": ("-o",),
    "javac": ("-d", "-s", "-h"),
    "sort": ("-o", "--output"),
    "ruff": ("--output-file",),
    "mypy": ("--junit-xml", "--cache-dir"),
    "pytest": ("--junitxml", "--junit-xml"),
    **{cmd: ("-output-directory", "--output-directory", "-jobname", "--jobname",
             "-output-format", "-aux-directory", "--aux-directory")
       for cmd in ("pdflatex", "latex", "xelatex", "lualatex", "pdftex", "tex",
                   "latexmk")},
    "makeindex": ("-o", "-t", "-l"),
    "bibtex": ("-o",),
    "biber": ("--outfile", "--output-file", "--output-directory"),
    "dvips": ("-o",),
    "dvipdf": ("-o",),
    "cmake": ("-B",),
}


def write_value_operands(argv: list[str]) -> list[str]:
    """Raw values of *argv*'s write flags, in either spelling.

    ``-o out.o`` (separated), ``-oout.o`` (glued short) and ``--output-file=out.txt``
    (glued long) all yield ``out.o``/``out.txt``. Returned as raw tokens like every
    other operand, for the caller to resolve and judge.
    """
    if not argv:
        return []
    flags = WRITE_VALUE_FLAGS_BY_CMD.get(argv[0], ())
    if not flags:
        return []
    out: list[str] = []
    i, n = 1, len(argv)
    while i < n:
        arg = argv[i]
        for flag in flags:
            if arg == flag:
                if i + 1 < n:
                    out.append(argv[i + 1])
                    i += 1
                break
            if arg.startswith(flag + "="):
                out.append(arg[len(flag) + 1:])
                break
            # Glued short flag ('-oout.o'). Long flags need the '=' form above: '--o'
            # is a prefix of unrelated long options, so gluing without '=' is not a
            # spelling any of these tools accept.
            if not flag.startswith("--") and arg.startswith(flag) and len(arg) > len(flag):
                out.append(arg[len(flag):])
                break
        i += 1
    return [t for t in out if t and not t.startswith("-")]


def path_operand_tokens(argv: list[str]) -> list[str]:
    """The arguments of *argv* in path position, command-family aware (a grep pattern is text).

    Conservative: a token that might be a path is kept; write-flag values
    (:data:`WRITE_VALUE_FLAGS_BY_CMD`) are included.
    """
    if not argv:
        return []
    head = argv[0]
    # A program named by path (``./a.out``, ``../tools/x.sh``) is itself a path this
    # call opens — the most consequential one, since it is what executes — so it gets
    # the same treatment as any other operand rather than being skipped as argv[0].
    leading_path = [head] if is_path_like_command(head) else []
    text_flags = _TEXT_VALUE_FLAGS.get(head, ())
    saw_text_flag = False
    positionals: list[str] = []
    files_from_flags: list[str] = []

    i, n = 1, len(argv)
    while i < n:
        arg = argv[i]
        if arg.startswith("-"):
            if arg in text_flags:
                saw_text_flag = True
                i += 2  # the value is the pattern/script
                continue
            if any(arg.startswith(f) and len(arg) > len(f) for f in text_flags):
                saw_text_flag = True  # glued form, e.g. -efoo / --regexp=foo
                i += 1
                continue
            if arg in _FILE_VALUE_FLAGS and i + 1 < n:
                saw_text_flag = True  # the pattern came from a file, not a positional
                files_from_flags.append(argv[i + 1])
                i += 2
                continue
            i += 1
            continue
        positionals.append(arg)
        i += 1

    if head in _LEADING_PATTERN_COMMANDS and not saw_text_flag and positionals:
        positionals = positionals[1:]
    written = [t for t in write_value_operands(argv) if t not in positionals]
    return leading_path + files_from_flags + written + positionals


def takes_path_operands(argv0: str) -> bool:
    """True when *argv0*'s arguments name files (the default); False for ``echo``, ``which``."""
    return bool(argv0) and argv0 not in PATH_INSENSITIVE_COMMANDS


def expansion_operands(argv: list[str]) -> list[str]:
    """Path-position tokens carrying a shell expansion, for classified commands only.

    They cannot be validated (expansion happens in the child's environment), so callers
    refuse them.
    """
    if not argv or argv[0] not in KNOWN_COMMANDS and not is_path_like_command(argv[0]):
        return []
    if not takes_path_operands(argv[0]):
        return []
    return [t for t in path_operand_tokens(argv) if "$" in t or "`" in t]


def segment_path_operands(argv: list[str], cwd: str) -> list[str]:
    """Resolved paths one command's arguments name, relative to *cwd*.

    Empty for a command that takes no file operand, so a caller can walk every
    segment of a chain uniformly. Tokens carrying an expansion are excluded — see
    :func:`expansion_operands`, which the server rejects on separately.
    """
    if not argv or not takes_path_operands(argv[0]):
        return []
    out: list[str] = []
    for arg in path_operand_tokens(argv):
        if "$" in arg or "`" in arg:
            continue
        resolved = normalize_path_arg(arg, cwd)
        if resolved is not None and resolved not in out:
            out.append(resolved)
    return out


def cd_destination(argv: list[str], cwd: str, call_cwd: str) -> str:
    """Where a ``cd`` segment moves the shell, resolved against *cwd*.

    Later relative paths resolve against it. A bare ``cd``/``~`` is the real ``$HOME``;
    ``cd -`` resets to *call_cwd*.
    """
    args = argv[1:]
    # '-' is tested before the operand scan below, which skips it as a flag.
    if "-" in args:
        return call_cwd
    target = next((a for a in args if not a.startswith("-")), None)
    if target is None or target == "~":
        return os.path.realpath(os.path.expanduser("~"))
    return normalize_path_arg(target, cwd) or call_cwd

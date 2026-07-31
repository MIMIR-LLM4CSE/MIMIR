"""How a shell command splits into commands, and which paths its arguments name.

Single source of truth for two halves of the same rule, which must not drift:

- the bash server's sandbox guard *confines* those paths to the workspace, and
- the client's out-of-workspace gate *prompts* the user for the ones outside it,
  so an approved path is then accepted by the guard above.

If the two disagreed the failure is silent and bad in both directions: a path the
client misses is refused by the server with no way for the user to grant it, and a
path the server misses is never gated at all. Hence one module, imported by both.

The *segmentation* below (:func:`parse_segments`) is here for the same reason: both
sides have to agree on what the shell will run — where one command ends and the
next begins, and which tokens are a redirection rather than an argument — before
either can say anything about the paths involved. Two tokenizers meant one could
read ``ls > out.txt`` as a write of ``out.txt`` while the other read ``out.txt`` as
an argument of ``ls``, and only one of them would ask the user about it.

Dependency-free (stdlib only) so it imports cleanly on either side of the
client/server process boundary — flat ``from shell_paths import ...`` in a server
subprocess, packaged ``mimir.servers._shared.shell_paths`` from the client
(cf. ``trusted_read_roots.py``).

Scope note: this module answers "which tokens are paths, and where do they
resolve" — nothing more. Whether a resolved path is *allowed* is the caller's
decision (the server checks workspace membership plus the user's session grants;
the client checks membership to decide whether to prompt).
"""

import os
import re
import shlex
from dataclasses import dataclass, field

# Build / execution — the compile/run/test/validate surface. Named as a group
# because it feeds both the bash server's allowlist and PATH_SENSITIVE_COMMANDS
# below: an interpreter or compiler takes a file operand exactly as ``cat`` does,
# so ``python /tmp/evil.py`` must be confined for the same reason
# ``cat /etc/passwd`` is — and it additionally *executes* what it reads.
EXEC_COMMANDS = frozenset({
    "python", "python3", "gcc", "g++", "gfortran", "javac", "java", "node", "pytest",
    # Python lint / typecheck / format tools, invocable directly by name (they also
    # work via ``python -m <tool>``). They replace the former dedicated code-quality
    # validator tools: the agent validates a written file with ``ruff check <f>``,
    # ``python -m mypy <f>``, ``pyflakes <f>`` or ``black <f>``.
    "ruff", "pyflakes", "mypy", "black",
    # GPU / build-system toolchain. ``nvcc`` is the CUDA compiler driver; ``make``
    # and ``cmake`` are how CUDA/HPC benchmarks usually drive a build; ``ctest``
    # runs a CMake project's test suite. A Makefile recipe is effectively arbitrary
    # execution, but no more so than the gcc/python already here.
    "nvcc", "make", "cmake", "ctest",
    # TeX/LaTeX document toolchain: a real execution that writes its artefacts
    # (.pdf/.aux/.log) next to the source, exactly as ``gcc -o`` does.
    "pdflatex", "latex", "xelatex", "lualatex", "pdftex", "tex",
    "bibtex", "biber", "makeindex", "latexmk", "dvips", "dvipdf",
})

# Environment managers: the package/env provisioning surface (``pip install``,
# ``conda create``). Kept apart from EXEC_COMMANDS, which means "compiles or runs the
# project's code" — these mutate the *interpreter environment* instead, which is a
# different kind of effect (the client classifies them ENV_*, and the bash server
# scopes their sub-commands the way it scopes ``module``'s).
#
# They still take file operands — a requirements file, a local wheel, an env YAML,
# ``pip install .`` — so they are path-sensitive like everything else. What they
# install lands wherever the target interpreter's site-packages is, which is outside
# the workspace by construction: that is the point of allowing them, and it is the
# approval prompt, not this module, that governs it.
ENV_MANAGER_COMMANDS = frozenset({
    "conda", "conda3", "mamba", "mamba3", "pip", "pip3",
})

# The remaining command groups, so the whole taxonomy of what a shell call can do
# lives in one module: the bash server builds its allowlist from these groups, and the
# client's classifier maps each to a capability kind. Split by *effect*, which is what
# both sides key off — read, search, inspect, write, execute, provision — because that
# is what decides approval and plan-mode availability.
#
# ``sed`` and ``sort`` sit in READ: they only write when carrying an in-place/output
# flag, which the client detects per call (a group cannot express "depends on a flag").
READ_COMMANDS = frozenset({
    "cat", "head", "tail", "nl", "sed", "wc", "cut", "sort", "uniq", "comm",
    "tr", "fold", "column", "cksum", "md5sum", "sha256sum",
    # ``stat``/``file`` open the file they are given, so they read it — unlike the
    # metadata no-ops below, whose arguments are words.
    "stat", "file",
})
SEARCH_COMMANDS = frozenset({"grep", "rg"})
INSPECT_COMMANDS = frozenset({"ls", "find", "du"})
# Unconditional writes: they move, duplicate or create whatever their flags say.
#
# ``chmod`` is a write of a file's *mode* rather than its contents, and it is here for
# one reason: without it a workspace script cannot be run at all. Executing one by path
# (``./build.sh``) is already supported — the bash server accepts a workspace-local
# executable — but a script arriving without the x bit (a fresh checkout, a file the
# agent just wrote) leaves no way to grant it, and the agent dead-ends on "Permission
# denied" with no allowed command that fixes it. It grants no new *kind* of power:
# ``python f.py``, ``make`` and a compiled ``./a.out`` already execute workspace code.
WRITE_COMMANDS = frozenset({"mv", "cp", "mkdir", "chmod"})
# Metadata and no-ops: no side effect, and no discovery evidence either. The no-ops are
# what makes a capability probe expressible end to end (``which pdflatex || true``).
NEUTRAL_COMMANDS = frozenset({
    "pwd", "echo", "which", "basename", "dirname", "realpath", "df",
    "true", "false", ":",
})

# Every command whose non-option arguments name files or directories — derived from the
# groups above rather than re-listed, so a command cannot be added to one and forgotten
# here. Reading, searching, inspecting, writing, executing and provisioning all take
# file operands, so all six are in.
#
# The two exceptions are what the derivation subtracts and adds: ``tr`` reads stdin only
# and its arguments are character SETS (``'a-z'``, ``'/'``), so confining them would
# reject a legitimate set; ``realpath`` is otherwise a no-op whose argument *is* a path.
# ``echo``/``which``/``pwd``/``df`` stay out — their arguments are words.
# What may appear as a nested command (``find … -exec CMD {} \;``). Derived from the
# read-only groups rather than re-listed, on the same principle as
# PATH_SENSITIVE_COMMANDS below: a command added to a group is covered automatically.
# Writers, execs and provisioners stay out — a nested command's operands include the
# ``{}`` placeholder, which no caller can resolve to the paths it will expand to, so
# a nested write is a write to somewhere unknowable. Reading one is harmless: the
# worst case is that some file inside the allowed roots is read.
READONLY_NESTED_COMMANDS = frozenset(
    READ_COMMANDS | SEARCH_COMMANDS | INSPECT_COMMANDS | NEUTRAL_COMMANDS
)

PATH_SENSITIVE_COMMANDS = frozenset(
    (READ_COMMANDS - {"tr"})
    | {"realpath"}
    | SEARCH_COMMANDS
    | INSPECT_COMMANDS
    | WRITE_COMMANDS
    | EXEC_COMMANDS
    | ENV_MANAGER_COMMANDS
)

# command → what running it does, for the one caller that needs to *report* the
# taxonomy rather than act on it (the server's allowlist introspection, so the agent can
# see what is plan-safe without probing). ``cd`` and the environment managers depend on
# their argument, hence their own labels.
COMMAND_CATEGORIES = {
    **{c: "read" for c in READ_COMMANDS},
    **{c: "search" for c in SEARCH_COMMANDS},
    **{c: "inspect" for c in INSPECT_COMMANDS},
    **{c: "write" for c in WRITE_COMMANDS},
    **{c: "exec" for c in EXEC_COMMANDS},
    **{c: "env" for c in ENV_MANAGER_COMMANDS},
    **{c: "neutral" for c in NEUTRAL_COMMANDS},
    "module": "env",
    "cd": "chdir",
}


_TRAILING_SEPARATOR = re.compile(r"(?:^|\s)(?:&&|\|\||\||;|&)$")
_LEADING_SEPARATOR = re.compile(r"^(?:&&|\|\||\||;)(?:\s|$)")


def flatten_unquoted_newlines(command: str) -> str:
    """Rewrite a multi-line command as the equivalent one-liner, for validation.

    A raw newline is a shell command separator, so a validator that tokenizes with
    ``shlex`` (where newline is mere whitespace) would read ``cat a\\nrm -rf b`` as
    *one* command ``cat`` with extra arguments — argv[0] passes the allowlist while
    bash actually runs both. That is why both validators used to refuse newlines
    outright. Refusing them also refuses the legitimate shape they usually carry: a
    multi-line ``python3 -c "..."`` body, or a chain written one command per line.

    So instead of rejecting, normalize: every **unquoted** newline becomes an
    explicit ``;`` and validation proceeds unchanged, which keeps the allowlist
    load-bearing per command. Newlines *inside* quotes are left exactly as they are
    — they are data (the ``-c`` body), not separators — and a backslash-newline is
    removed, as the shell does. No ``;`` is inserted where the line already ends (or
    the next begins) with a separator, so ``make &&\\n./a.out`` stays valid.

    The string returned is for the *checker* only; the caller still hands bash the
    original text, whose newlines bash reads as the separators this models.
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
    """The first substitution marker *the shell would act on*, or None.

    Quoting decides whether these characters run anything, so the test has to be
    made where the quoting is still visible — on the raw string. The token scan this
    replaces ran after ``shlex(posix=True)`` had already **removed** the quotes, so a
    backtick that bash treats as text was indistinguishable from one that executes,
    and every literal occurrence was refused: ``sed -n '/```mermaid/,/```/p' f.md``
    and ``grep -n '${VAR}' f.py`` are searches for a *string*, and are exactly where
    these characters legitimately appear.

    Single quotes make everything literal, so markers inside them are text. Double
    quotes do **not** — ``"`cmd`"`` and ``"$(cmd)"`` both substitute — so those still
    count, as does anything unquoted. A backslash-escaped marker outside single
    quotes is literal too, and is skipped the way the shell skips it.
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
                if command.startswith(marker, i):
                    return marker
        i += 1
    return None


# ── Segmentation: one shell string → the commands the shell would run ──────────

# Separators that join several commands in one call. Each command between them is
# a separate argv, validated (server) and classified (client) on its own.
COMMAND_SEPARATORS = frozenset({";", "&&", "||", "|"})

# Markers that run or inline *another* command. One of these is refused outright:
# whatever the allowlist said about argv[0] tells us nothing about what the
# substitution would execute.
#
# "Contains a marker" is only the right test where the shell would *act* on it — see
# :func:`unquoted_substitution_marker`. Inside single quotes these characters are
# ordinary text, and a search pattern is the usual place they show up.
SUBSTITUTION_MARKERS = ("$(", "`", "${", "<(", ">(")

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
    """One command of a chain: its argv, plus the files its redirections touch.

    ``nested`` marks a segment lifted out of another command's argument list rather
    than found between shell separators — today only a ``find -exec`` payload. It is
    a real command the shell will run, so it is emitted as its own segment and every
    caller confines its operands and classifies its head exactly as usual; the flag
    only lets the *policy* layer treat it differently from a top-level command.
    """
    argv: list[str]
    redirections: list[Redirection] = field(default_factory=list)
    nested: bool = False

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


def parse_segments(command: str) -> list[ParsedSegment]:
    """Split *command* into the commands the shell would run, with their redirections.

    Quote-aware: chaining separators and other operators become standalone tokens,
    while anything inside quotes is preserved literally (a ``;`` inside ``python -c
    "a; b"`` is text, not chaining). Multi-line input is normalized first, so an
    unquoted newline separates commands exactly as ``;`` does — see
    :func:`flatten_unquoted_newlines`.

    Redirections are lifted out of argv and attached to their segment: fd dups and
    ``/dev/null`` vanish (they name no file), and ``> f`` / ``>> f`` / ``< f`` are
    recorded as a :class:`Redirection` so the caller can treat ``f`` as the path it
    is — confined to the workspace by the server, prompted for by the client, and
    counted as a write by the classifier. The leading source fd (the ``2`` in
    ``2>/dev/null``) is dropped from argv rather than left to look like an argument.

    Raises :class:`ShellParseError` for what neither side will reason about:
    substitution, a subshell, backgrounding, a heredoc, a dangling separator or a
    redirection with no target. Everything else — globs, ``$VAR``, unknown commands
    — parses; judging those is the caller's job, not this function's.
    """
    if not isinstance(command, str) or not command.strip():
        raise ShellParseError("empty", "Command must be a non-empty string.")

    command = flatten_unquoted_newlines(command)
    # Before tokenizing, while the quotes are still there to read: shlex strips them,
    # so this cannot be decided per token afterwards.
    marker = unquoted_substitution_marker(command)
    if marker is not None:
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

    segments: list[ParsedSegment] = []
    argv: list[str] = []
    redirections: list[Redirection] = []
    # A `find -exec` payload: tokens after the flag belong to a nested command, not
    # to find's own argv, and the `;`/`+` that ends it is a terminator rather than a
    # chaining separator. Without this the terminator would split the command in the
    # wrong place and the nested argv would be indistinguishable from find's operands.
    nested_argv: list[str] | None = None
    nested_segments: list[ParsedSegment] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]

        if nested_argv is not None:
            if tok in _EXEC_TERMINATORS:
                if nested_argv:
                    nested_segments.append(ParsedSegment(nested_argv, [], nested=True))
                nested_argv = None
                i += 1
                continue
            nested_argv.append(tok)
            i += 1
            continue

        if tok in _NESTED_COMMAND_FLAGS and argv:
            # The flag itself is dropped from the outer argv: it is not an operand,
            # and leaving it there would trip the server's per-command flag denylist
            # on the very token whose payload has just been made inspectable. The
            # resulting nested segment is what records that a nested command exists.
            nested_argv = []
            i += 1
            continue

        if tok in COMMAND_SEPARATORS:
            if not argv:
                raise ShellParseError(
                    "separator", "A command separator must join two real commands.")
            segments.append(ParsedSegment(argv, redirections))
            argv, redirections = [], []
            i += 1
            continue

        is_operator = bool(tok) and all(ch in _OPERATOR_CHARS for ch in tok)
        if is_operator and ("<" in tok or ">" in tok):
            if tok.startswith("<<"):
                # Heredoc / herestring: its body is not an argv this can segment,
                # and for '<<' the body lines would each be read as a command.
                raise ShellParseError(
                    "heredoc", f"Redirection operator '{tok}' is not supported.")
            # Quote-aware tokenization detaches the source fd (the '2' in
            # '2>/dev/null') into its own token; drop it so it never lands in argv.
            if argv and len(argv[-1]) == 1 and argv[-1].isdigit():
                argv.pop()
            target = tokens[i + 1] if i + 1 < n else None
            if target is None or target in COMMAND_SEPARATORS or all(
                ch in _OPERATOR_CHARS for ch in target
            ):
                raise ShellParseError("redirect", "Redirection is missing a target.")
            # A pure fd reference ('1', '2-', '-') or the null sink names no file.
            if not (target == "-" or target.rstrip("-").isdigit() or target in _FD_SINKS):
                redirections.append(Redirection(tok, target, writes=">" in tok))
            i += 2  # the operator and its target
            continue

        if is_operator:
            raise ShellParseError(
                "operator", f"Shell operator '{tok}' is not allowed.")

        argv.append(tok)
        i += 1

    if nested_argv is not None:
        raise ShellParseError(
            "separator",
            "A nested command (-exec/-execdir) must be terminated by ';' or '+'.",
        )
    if argv:
        segments.append(ParsedSegment(argv, redirections))
    elif redirections:
        raise ShellParseError("redirect", "A redirection must belong to a command.")
    if not segments:
        raise ShellParseError("empty", "Invalid command syntax.")
    # Nested commands trail the chain they were lifted from: order within the
    # top-level chain is preserved for every caller that reasons about `cd` rebasing,
    # and a nested command can never precede the command that invokes it anyway.
    return segments + nested_segments


def is_path_like_command(argv0: str) -> bool:
    """True if *argv0* references a program by path (``./a.out``) rather than name.

    Existence is intentionally not checked: validation runs before a chained
    compile step has produced the binary. Such a command executes whatever it
    points at, so its own operands are path-sensitive too.
    """
    return bool(argv0) and (os.sep in argv0 or argv0.startswith("." + os.sep))


def normalize_path_arg(raw_arg: str, cwd: str) -> str | None:
    """Resolve a path-like argument to a real path, or None if it is not one.

    Returns None for the tokens that are *not* paths — option flags, ``-`` (stdin),
    URLs — so callers skip them. Everything else resolves against *cwd*, which is
    what makes ``-I/usr/include`` and ``-lm`` pass untouched while a bare
    ``/etc/passwd`` or ``../../x.py`` resolves to something the caller can judge.
    """
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

# Flags whose value is a file the command *writes*, keyed by leading command.
#
# Why these and not flags generally: a flag value is skipped by the walk below, which
# is right for the ones that carry no path (`-lm`) and for the ones that read outside
# the workspace legitimately — `-I/usr/include`, `-L/usr/lib` and `-Wl,-rpath,...` are
# how any real build finds its system headers and libraries, and confining them would
# refuse every HPC compile. Writing outside the workspace is the other case entirely:
# it needs the user's approval, and the *separated* form already got it (`gcc -o
# /tmp/x.out` resolves as a positional and is refused). The glued form did not —
# `gcc a.c -o/tmp/x.out`, `ruff check --output-file=/tmp/r.txt`,
# `pytest --junitxml=/tmp/j.xml` all wrote outside with no prompt — which is the
# asymmetry this table closes. Both spellings of a *write* flag yield their value.
#
# Add a flag here when its value is a path the command creates or overwrites. Do not
# add read/search paths: they are deliberately unconfined (see above).
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
    """The raw arguments of *argv* that sit in path position.

    Command-family aware, because "every non-flag argument is a path" is wrong for
    the two families that take text first: a `grep` pattern and a `sed` script are
    data, not files. Treating them as paths made `grep /etc/passwd notes.txt` — a
    search for a *string* — look like a read of /etc/passwd, so it was refused and
    (once the client gate existed) prompted the user about a file nothing would open.

    Conservative in the safe direction: a token that might be a path is kept. Flag
    values are only skipped when the flag is known to take text — except for the
    *write* flags (``gcc -o``, ``ruff --output-file``, …), whose value is a file the
    command creates and is therefore included in either spelling: see
    :data:`WRITE_VALUE_FLAGS_BY_CMD` for why writes are treated differently from the
    ``-I``/``-L`` reads that stay unconfined.
    """
    if not argv:
        return []
    head = argv[0]
    # A program named by path (``./a.out``, ``../tools/x.sh``) is itself a path this
    # call opens — the most consequential one, since it is what executes. Walking only
    # argv[1:] left it unexamined by both sides: the client never prompted for it, and
    # the server could only refuse it outright as an unknown command, so an executable
    # outside the workspace had no route to approval while a *file* outside it did.
    # Including it here gives it the same treatment as any other operand.
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
    """True when *argv0*'s arguments name files/dirs, so they must be checked.

    ``cd`` counts: its target is a directory the caller resolves. ``echo``/``which``
    do not — their arguments are words, and treating them as paths would refuse
    ``echo $HOME`` for no reason.
    """
    return (argv0 in PATH_SENSITIVE_COMMANDS
            or argv0 == "cd"
            or is_path_like_command(argv0))


def expansion_operands(argv: list[str]) -> list[str]:
    """Path-position tokens carrying a shell expansion (``$VAR``, backticks).

    These cannot be validated: expansion happens in the *child* shell, whose
    environment is not the one doing the checking — and a chain can change it
    mid-flight. ``module load cuda && cat $CUDA_HOME/version.txt`` checked as a
    harmless relative path (CUDA_HOME unset here) and then read from the module
    tree at runtime. The command that runs must be the command that was checked,
    so a path built from an expansion is refused rather than guessed at.

    Only path position, and only for commands that open paths at all:
    ``-I$CUDA_HOME/include`` is a flag and ``echo $HOME`` opens nothing, so both
    keep working — the HPC idioms are untouched.
    """
    if not argv or not takes_path_operands(argv[0]):
        return []
    return [t for t in path_operand_tokens(argv) if "$" in t or "`" in t]


def segment_path_operands(argv: list[str], cwd: str) -> list[str]:
    """Resolved paths one command's arguments name, relative to *cwd*.

    Empty for a command that takes no file operand, so a caller can walk every
    segment of a chain uniformly. Tokens carrying an expansion are excluded — see
    :func:`expansion_operands`, which the server rejects on separately.
    """
    if not argv:
        return []
    if argv[0] not in PATH_SENSITIVE_COMMANDS and not is_path_like_command(argv[0]):
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

    The directory a chain lands in is the base every later relative path resolves
    against, so a caller walking a chain must thread this value through — checking
    each segment against the *initial* directory instead would make both halves of
    ``cd /etc && cat passwd`` look individually fine.

    A bare ``cd`` and ``cd ~`` go to ``$HOME``, which is the user's real home: the
    shell inherits it rather than having it pinned to the workspace, so this must
    resolve there too. It is then judged like any other destination — outside the
    workspace, hence refused unless the user approved it — instead of being modelled
    as a return to the workspace, which would have let the rest of the chain resolve
    its relative paths against a directory the shell had already left.

    ``cd -`` is the one target that cannot be read off the command: it is ``$OLDPWD``,
    which depends on where the chain has been. It resets to *call_cwd*, which is safe
    because OLDPWD can only ever be a directory this same chain already occupied, and
    every one of those passed the caller's confinement check to be entered at all.
    """
    args = argv[1:]
    # '-' is tested before the operand scan below, which skips it as a flag.
    if "-" in args:
        return call_cwd
    target = next((a for a in args if not a.startswith("-")), None)
    if target is None or target == "~":
        return os.path.realpath(os.path.expanduser("~"))
    return normalize_path_arg(target, cwd) or call_cwd

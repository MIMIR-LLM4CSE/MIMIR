"""Classify a ``bash_run`` command into capability kinds and the operands they act on.

Segments come from ``shell_paths.parse_segments``, the parser the server validates with.
Anything ambiguous yields ``None`` (opaque). This observes intent; the bash server is the
security boundary.
"""

from __future__ import annotations

import os
import re
from typing import NamedTuple

# Tokenization, command groups and path-operand extraction all come from the module
# the bash *server* validates with, so classification tracks what the server
# actually permits and confines — one implementation, no drift (see its docstring).
from ...shared.shell_paths import (
    COMMAND_WRAPPERS,
    EFFECT_RUN,
    allocates_cluster,
    ENV_MANAGER_COMMANDS,
    EXEC_COMMANDS,
    EXEC_EFFECTS,
    INSPECT_COMMANDS,
    GIT_READ_SUBCOMMANDS,
    GIT_VALUE_FLAGS,
    NEUTRAL_COMMANDS,
    READ_COMMANDS,
    SEARCH_COMMANDS,
    WRITE_COMMANDS,
    ParsedSegment,
    ShellParseError,
    is_path_like_command,
    parse_segments,
    unquoted_shell_char,
    unwrap_argv,
    path_operand_tokens,
    write_value_operands,
)


# ── Capability kinds a bash segment can carry ──────────────────────────────────
class Kind:
    READ = "read"
    SEARCH = "search"
    INSPECT = "inspect"
    WRITE = "write"
    EXEC = "exec"
    ENV_DISCOVERY = "env_discovery"
    ENV_MUTATE = "env_mutate"
    CHDIR = "chdir"
    NEUTRAL = "neutral"
    # A head no group places. Runnable (bash gates by denylist), so it must classify —
    # as an execution, which is the conservative reading: never plan-safe, never
    # approval-exempt, and owing a verdict like any other run.
    UNKNOWN = "unknown"


# Kinds with no filesystem/exec side effect — what makes a command "plan-safe"
# (runnable while drafting / auto-approvable as read-only). CHDIR belongs here: a bare
# ``cd`` only moves the shell's cwd for the rest of the one command.
READONLY_KINDS = frozenset({
    Kind.READ, Kind.SEARCH, Kind.INSPECT, Kind.ENV_DISCOVERY, Kind.CHDIR, Kind.NEUTRAL,
})


class Segment(NamedTuple):
    kind: str
    operands: list[str]  # files/dirs/patterns the segment acts on (may be empty)
    head: str = ""       # leading command of an EXEC segment (module for `python -m X`), else ""
    argv: tuple[str, ...] = ()  # the segment's raw tokens; operands drop the flags
    # What an EXEC segment's success would *prove* (see shell_paths.EXEC_EFFECTS).
    # Deliberately not a Kind: policy, plan mode and approval treat all four alike,
    # and only the evidence layer tells a check from a build from a run.
    effect: str = ""


# Expansions whose value only the running shell knows, making a command's kind a guess.
# Read off the raw command (``unquoted_shell_char``), never off tokens: shlex strips the
# quotes that tell ``grep '$HOME'`` from an expansion.
_EXPANSION_SIGILS = frozenset("$")

# Leading command → kind comes from the shell_paths command *groups*, the same module
# the bash server validates with, so classification cannot drift from what runs.
# This module owns what a group cannot express: which option or sub-command changes the
# effect (`sed -i` writes, `module load` mutates, `> file` turns a read into a write).
# `cd` gets its own kind so its target can rebase later segments' relative operands.

# Sub-commands that only *report* what is installed, offline: discovery, hence
# plan-safe. Everything else the server allows (install/create/update, plus the network
# queries search/index/download) is ENV_MUTATE — not because each writes, but because
# none is a local read, and a plan is drafted without reaching the network.
_ENV_MANAGER_QUERY_SUB = frozenset({
    "list", "show", "freeze", "check", "inspect", "info", "help",
})

# `module` sub-commands: discovery (read-only) vs env-mutating load.
_MODULE_DISCOVERY_SUB = frozenset({
    "list", "avail", "av", "show", "display", "spider", "whatis",
    "help", "keyword", "overview", "is-loaded", "is-avail",
})
_MODULE_MUTATE_SUB = frozenset({"load", "add"})

# In-place / output-write flags that turn an otherwise read-only command into a
# write, keyed by leading command. `sed -i` rewrites files in place; `sort -o FILE`
# writes its output. Kept here (not plan_mode) so classification is the one source.
_WRITE_FLAGS_BY_CMD = {
    "sed": ("-i", "--in-place"),
    "sort": ("-o", "--output"),
}


def _has_write_flag(argv: list[str]) -> bool:
    """True when *argv* carries an in-place/output-write flag for its command.

    Short flags may be glued (``sed -i.bak``) or clustered (``sed -ni``), and the
    long form may take an ``=value`` (``sort --output=x``); all count as a write.
    Conservative: a spurious match only reclassifies the segment as a write.
    """
    write_flags = _WRITE_FLAGS_BY_CMD.get(argv[0])
    if not write_flags:
        return False
    for arg in argv[1:]:
        for flag in write_flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
            # Glued/clustered short flag, e.g. '-i.bak', '-ni', '-o' variants.
            if len(flag) == 2 and flag[1].isalpha():
                if arg.startswith("-") and not arg.startswith("--") and flag[1] in arg:
                    return True
    return False


# ── Arguments that make a read run a program or write a file ───────────────────
# A read-only head is read-only only with arguments that keep it so: GNU sed's ``e``
# runs a command and ``w`` writes a file, ``git -c core.fsmonitor=…`` runs one, ``find
# -delete`` deletes. Anything this section cannot vouch for reads as a run, which is
# what the approval exemption and plan mode rely on.

# sed commands taking no argument, and those whose argument is a label or a number.
_SED_PLAIN = frozenset("=dDgGhHlnNpPxzF")
_SED_LABELLED = frozenset("btT:")
_SED_NUMBERED = frozenset("qQlL")
# Commands that run a program, or touch a file other than the input.
_SED_UNSAFE = frozenset("ewWrRv")
# Flags of ``s///`` that only change what is matched or printed.
_SED_SAFE_S_FLAGS = frozenset("gpiImM0123456789")
_SED_VALUE_FLAGS = frozenset({"-l", "--line-length", "-e", "--expression"})


def _sed_regex_end(script: str, i: int, delim: str) -> int:
    """The index just past the regex starting at *i*, delimited by *delim*; -1 if unterminated.

    Only a backslash escapes, as in GNU sed: a delimiter inside brackets still ends it.
    """
    n = len(script)
    while i < n:
        c = script[i]
        if c == "\\":
            i += 2
            continue
        if c == "\n" or c == delim:
            return i + 1 if c == delim else -1
        i += 1
    return -1


def _sed_address_end(script: str, i: int) -> int:
    """The index past an address at *i* (none: *i*); -1 when it cannot be read."""
    n = len(script)
    if i < n and (script[i].isdigit() or script[i] == "$"):
        i += 1
        while i < n and (script[i].isdigit() or script[i] == "~"):
            i += 1
        return i
    if i < n and script[i] in "/\\":
        if script[i] == "\\":
            if i + 1 >= n:
                return -1
            delim, i = script[i + 1], i + 2
        else:
            delim, i = "/", i + 1
        i = _sed_regex_end(script, i, delim)
        while 0 <= i < n and script[i] in "IM":
            i += 1
        return i
    if i < n and script[i] in "+~":
        i += 1
        while i < n and script[i].isdigit():
            i += 1
    return i


def _sed_script_is_readonly(script: str) -> bool:
    """Whether a sed *script* only prints: no ``e``, ``w``, ``r`` or ``s///e``/``w``.

    Parsed as GNU sed reads it, command by command. A script this cannot read is not
    read-only.
    """
    i, n = 0, len(script)
    while i < n:
        c = script[i]
        if c in " \t\n;{}!":
            i += 1
            continue
        if c == "#":
            while i < n and script[i] != "\n":
                i += 1
            continue
        i = _sed_address_end(script, i)
        if i < 0:
            return False
        if i < n and script[i] == ",":
            i = _sed_address_end(script, i + 1)
            if i < 0:
                return False
        while i < n and script[i] in " \t!":
            i += 1
        if i >= n:
            return False
        c = script[i]
        i += 1
        if c in _SED_UNSAFE:
            return False
        if c in _SED_PLAIN or c in "{}":
            continue
        if c in _SED_LABELLED or c in "aic":
            # A label, a number or text: the rest of the command, harmless as words.
            stop = "\n" if c in "aic:" else ";\n"
            while i < n and script[i] not in stop:
                i += 1
            continue
        if c in _SED_NUMBERED:
            while i < n and (script[i].isdigit() or script[i] in " \t"):
                i += 1
            continue
        if c in "sy":
            if i >= n or script[i] in "\n\\":
                return False
            delim = script[i]
            i = _sed_regex_end(script, i + 1, delim)
            if i < 0:
                return False
            i = _sed_regex_end(script, i, delim)
            if i < 0:
                return False
            if c == "s":
                while i < n and script[i] not in " \t\n;}":
                    if script[i] not in _SED_SAFE_S_FLAGS:
                        return False
                    i += 1
            continue
        return False
    return True


def _sed_is_readonly(argv: list[str]) -> bool:
    """Whether a ``sed`` call only reads: every script it is given only prints."""
    scripts: list[str] = []
    positional: list[str] = []
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in ("-f", "--file") or arg.startswith("--file=") or (
                arg.startswith("-f") and not arg.startswith("--")):
            return False
        if arg in ("-e", "--expression"):
            if i + 1 >= len(argv):
                return False
            scripts.append(argv[i + 1])
            i += 2
            continue
        if arg.startswith("--expression="):
            scripts.append(arg.split("=", 1)[1])
        elif arg.startswith("-e") and not arg.startswith("--"):
            scripts.append(arg[2:])
        elif arg in _SED_VALUE_FLAGS:
            i += 2
            continue
        elif arg == "--" or not arg.startswith("-") or arg == "-":
            positional.append(arg)
        elif not arg.startswith("--") and "e" in arg[1:]:
            # A cluster whose last letter takes the script: ``-ne '1e id'``.
            return False
        i += 1
    if not scripts:
        if not positional:
            return False
        scripts = [positional[0]]
    return all(_sed_script_is_readonly(script) for script in scripts)


# git's global options that bring configuration, another repository or another program
# with them. ``-C`` is not among them: its directory is a path operand, confined.
_GIT_UNSAFE_GLOBALS = ("-c", "--config-env", "--exec-path", "--git-dir", "--work-tree")
# Options of a reading subcommand that run a program or write a file.
_GIT_UNSAFE_OPTIONS = ("--output", "-O", "--open-files-in-pager", "--ext-diff",
                       "--textconv")
# The subcommands that list by default but create, delete or rewrite with an argument.
_GIT_LISTING_ONLY = frozenset({"branch", "tag", "remote", "reflog"})
_GIT_LIST_VALUE_FLAGS = frozenset({"--contains", "--no-contains", "--merged",
                                   "--no-merged", "--points-at", "--sort", "--format"})
_GIT_LIST_FLAGS = frozenset({"-l", "--list"})
_GIT_REMOTE_READS = frozenset({"show", "get-url"})
_GIT_REFLOG_WRITES = frozenset({"expire", "delete"})


def _git_option_matches(arg: str, options: tuple[str, ...]) -> bool:
    return any(arg == o or arg.startswith(o + "=")
               or (len(o) == 2 and arg.startswith(o) and not arg.startswith("--"))
               for o in options)


def _git_lists_only(sub: str, args: list[str]) -> bool:
    """Whether a ``branch``/``tag``/``remote``/``reflog`` call only lists."""
    positionals: list[str] = []
    listing = False
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in _GIT_LIST_VALUE_FLAGS:
            i += 2
            continue
        if arg in _GIT_LIST_FLAGS:
            listing = True
        elif arg.startswith("-"):
            if sub in ("branch", "tag") and arg not in (
                    "-a", "--all", "-r", "--remotes", "-v", "-vv", "--verbose",
                    "--show-current", "--column", "--no-column", "-i", "--ignore-case",
                    "--omit-empty") and not arg.startswith(("--sort=", "--format=",
                    "--contains=", "--no-contains=", "--merged=", "--no-merged=",
                    "--points-at=", "--color", "--no-color", "-n")):
                return False
        else:
            positionals.append(arg)
        i += 1
    if sub == "remote":
        return not positionals or positionals[0] in _GIT_REMOTE_READS
    if sub == "reflog":
        return not positionals or positionals[0] not in _GIT_REFLOG_WRITES
    # A name given to branch or tag creates one, unless the call lists by pattern.
    return not positionals or listing


def _git_is_readonly(argv: list[str], sub: str) -> bool:
    """Whether a reading ``git`` subcommand stays one with these arguments."""
    try:
        at = argv.index(sub, 1)
    except ValueError:
        return False
    if any(_git_option_matches(a, _GIT_UNSAFE_GLOBALS) for a in argv[1:at]):
        return False
    args = argv[at + 1:]
    if any(_git_option_matches(a, _GIT_UNSAFE_OPTIONS) for a in args):
        return False
    if sub in _GIT_LISTING_ONLY:
        return _git_lists_only(sub, args)
    return True


# What an ``env`` wrapper may set and still leave a read a read: locale and display.
# Anything else may load code (LD_PRELOAD), name a program (GIT_EXTERNAL_DIFF, PAGER)
# or change what the command reads.
_ENV_SAFE_NAMES = re.compile(r"^(?:LC_[A-Z]+|LANG|LANGUAGE|TZ|COLUMNS|NO_COLOR|TERM)$")
# The heads ``xargs`` may run and stay read-only: their options cannot run a program,
# whatever the input appends to them.
_XARGS_SAFE_HEADS = frozenset({
    "cat", "head", "tail", "wc", "nl", "ls", "stat", "file", "du", "grep",
    "md5sum", "sha256sum", "cksum",
})


def _wrappers_keep_readonly(argv: list[str], inner: list[str], wrappers: list[str]) -> bool:
    """Whether the wrappers around *inner* leave its kind unchanged."""
    names = {os.path.basename(w) for w in wrappers}
    if "xargs" in names and inner and inner[0] not in _XARGS_SAFE_HEADS:
        return False
    if "env" not in names:
        return True
    prefix = argv[:len(argv) - len(inner)]
    for tok in prefix:
        if tok in ("-S", "--split-string") or tok.startswith(("-S", "--split-string=")):
            return False
        if ENV_ASSIGN_RE.match(tok) and not _ENV_SAFE_NAMES.match(tok.split("=", 1)[0]):
            return False
    return True


def _find_is_readonly(argv: list[str]) -> bool:
    """``find -delete`` deletes; nested ``-exec`` commands are judged as segments of their own."""
    return "-delete" not in argv[1:]


def _readonly_arguments(argv: list[str]) -> bool:
    """Whether *argv*'s arguments keep its read-only head read-only."""
    head = argv[0]
    if any(a == "--output" or a.startswith("--output=") for a in argv[1:]):
        return False
    if head == "sed":
        return _sed_is_readonly(argv)
    if head == "sort":
        return not any(a.startswith("--compress-program") for a in argv[1:])
    if head == "find":
        return _find_is_readonly(argv)
    return True


def _looks_like_path(token: str) -> bool:
    """Heuristic: a positional that plausibly names a file/dir, not an option/value.

    Accepts things containing a path separator or a dotted extension, plus bare
    names — but rejects option flags and numeric option-values. Conservative: used
    only to *credit* operands, never to grant access (the server confines paths).
    """
    if not token or token.startswith("-"):
        return False
    if token.isdigit():
        return False
    return True


def _write_operand(argv: list[str]) -> list[str]:
    """The file a `sed -i` / `sort -o` segment writes.

    `sort -o FILE`: the flag's value, in whichever spelling — the same extraction the
    server confines it with (``shell_paths.write_value_operands``).
    `sed -i SCRIPT FILE...`: the file positionals (script skipped, see _read_operands).
    """
    written = write_value_operands(argv)
    if written:
        return [t for t in written if _looks_like_path(t)]
    # sed -i: operands are the file positionals (same extraction as a read).
    return _read_operands(argv)


def _destination_operands(argv: list[str]) -> list[str]:
    """The path an `mv`/`cp`/`mkdir` segment writes to.

    Only the destination is credited: after `mv a b` the source no longer exists.
    """
    positionals = [a for a in argv[1:] if _looks_like_path(a)]
    if not positionals:
        return []
    if argv[0] == "mkdir":
        return positionals
    return positionals[-1:]


# Read commands that take NO file operand — they consume stdin only, and their
# positionals are data (character SETs for `tr`), never paths. Extracting operands
# for these would credit bogus reads, so they classify as READ with no operand.
_STDIN_ONLY_READ = frozenset({"tr"})


def _file_operands(argv: list[str]) -> list[str]:
    """The path-position arguments of *argv*, from ``shell_paths.path_operand_tokens``.

    The server confines that same extraction; only tokens that cannot be paths (options,
    numeric flag values) are dropped here.
    """
    return [t for t in path_operand_tokens(argv) if _looks_like_path(t)]


def _read_operands(argv: list[str]) -> list[str]:
    """Path positionals a read command acts on (stdin-only readers have none)."""
    if argv[0] in _STDIN_ONLY_READ:
        return []
    return _file_operands(argv)


def _search_patterns(argv: list[str]) -> list[str]:
    """The pattern(s) a grep/rg segment searches for: ``-e`` values, else the first positional."""
    for i, arg in enumerate(argv[1:], start=1):
        if arg == "-e" and i + 1 < len(argv):
            return [argv[i + 1]]
        if arg.startswith("-e"):
            return [arg[2:]]
    positionals = [a for a in argv[1:] if _looks_like_path(a)]
    return positionals[:1]


def _inspect_operands(argv: list[str]) -> list[str]:
    """Directory operands for ls/find/du (first path positional; default '.')."""
    operands = _file_operands(argv)
    return operands[:1] if operands else ["."]


# A positional that unambiguously names a file: path separator or dotted extension.
# Stricter than :func:`_looks_like_path`, so a module name (`python -m py_compile`),
# sub-command (`ruff check`) or bare flag value is never mistaken for the target file.
# A shell ``VAR=value`` assignment word.
ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")
# A token that names a file by its extension.
FILE_LIKE_RE = re.compile(r"\.[A-Za-z0-9]+$")


def _exec_head(argv: list[str]) -> str:
    """The leading command of an exec segment; for ``python -m X``, the module ``X``."""
    head = argv[0]
    if head in ("python", "python3"):
        i = 1
        while i < len(argv):
            a = argv[i]
            if a == "-m" and i + 1 < len(argv):
                return argv[i + 1]
            if not a.startswith("-"):
                break  # first positional is a script path, not a -m module
            i += 1
    return head


def _exec_operands(argv: list[str]) -> list[str]:
    """File positionals an exec or validation command acts on (best-effort).

    Only tokens that clearly name a file (a ``/`` or an extension) are credited, so
    ``pytest test_foo.py`` marks ``test_foo.py`` validated.
    """
    return [
        a for a in argv[1:]
        if not a.startswith("-") and ("/" in a or FILE_LIKE_RE.search(a))
    ]


def _module_segment(argv: list[str]) -> Segment:
    sub = next((a for a in argv[1:] if not a.startswith("-")), None)
    names = [a for a in argv[2:] if not a.startswith("-")] if sub else []
    if sub in _MODULE_MUTATE_SUB:
        return Segment(Kind.ENV_MUTATE, names)
    if sub in _MODULE_DISCOVERY_SUB:
        return Segment(Kind.ENV_DISCOVERY, [])
    # Unknown/absent sub-command: assume it mutates, as for the other environment
    # managers (:func:`_env_manager_segment`). Guessing "harmless" here would make a
    # sub-command nobody has reasoned about plan-safe; the server refuses it anyway.
    return Segment(Kind.ENV_MUTATE, names)


def _env_manager_segment(argv: list[str]) -> Segment:
    """Classify a pip/conda/mamba call: a local query is discovery, the rest mutates.

    The operands of a mutation are the names it brings in (packages, or the env being
    created) — the same shape ``module load``'s are, so the env-mutation record reads
    the same whichever provisioning route the model took.
    """
    positionals = [a for a in argv[1:] if not a.startswith("-")]
    sub = positionals[0] if positionals else None
    if sub in _ENV_MANAGER_QUERY_SUB:
        return Segment(Kind.ENV_DISCOVERY, [])
    # `conda env create …` nests the verb one level deeper, so the names start after it.
    names = positionals[2:] if sub == "env" else positionals[1:]
    return Segment(Kind.ENV_MUTATE, names)


def _classify_segment(argv: list[str]) -> Segment:
    """Map one command's argv to a Segment. Never None: an unplaced head is UNKNOWN."""
    # `timeout 60 pytest -q` is a pytest run: the wrapper is stripped before anything
    # else reads the head, so kind, effect and operands are the inner command's.
    raw = argv
    argv, wrappers = unwrap_argv(argv)
    head = argv[0]
    if not _wrappers_keep_readonly(raw, argv, wrappers):
        return Segment(Kind.UNKNOWN, _file_operands(argv), head, effect=EFFECT_RUN)

    # A workspace-local executable (./a.out) is an execution. Same test the server
    # uses to decide an argv0 names a program by path rather than by name.
    if is_path_like_command(head):
        return Segment(Kind.EXEC, _exec_operands(argv), head, effect=EFFECT_RUN)

    if head == "cd":
        # Target dir (first positional; `cd`, `cd -`, `cd ~` carry none we can rebase).
        target = next((a for a in argv[1:] if not a.startswith("-") and a != "~"), None)
        return Segment(Kind.CHDIR, [target] if target else [])

    if head == "git":
        return _git_segment(argv)
    if head == "module":
        return _module_segment(argv)
    if head in ENV_MANAGER_COMMANDS:
        return _env_manager_segment(argv)
    if head in EXEC_COMMANDS:
        exec_head = _exec_head(argv)
        # An unknown ``python -m <module>`` resolves to no effect of its own; it runs
        # whatever the module does, which is the interpreter's own effect.
        effect = EXEC_EFFECTS.get(exec_head) or EXEC_EFFECTS.get(head, "")
        return Segment(Kind.EXEC, _exec_operands(argv), exec_head, effect=effect)
    if head in WRITE_COMMANDS:
        return Segment(Kind.WRITE, _destination_operands(argv))
    if head in READ_COMMANDS and _has_write_flag(argv):
        return Segment(Kind.WRITE, _write_operand(argv))
    if (head in SEARCH_COMMANDS or head in INSPECT_COMMANDS or head in READ_COMMANDS) \
            and not _readonly_arguments(argv):
        return Segment(Kind.UNKNOWN, _file_operands(argv), head, effect=EFFECT_RUN)
    if head in SEARCH_COMMANDS:
        return Segment(Kind.SEARCH, _search_patterns(argv))
    if head in INSPECT_COMMANDS:
        return Segment(Kind.INSPECT, _inspect_operands(argv))
    if head in READ_COMMANDS:
        return Segment(Kind.READ, _read_operands(argv))
    if head in NEUTRAL_COMMANDS:
        return Segment(Kind.NEUTRAL, [])
    # Unplaced head: read as a run. Its path-position operands are still credited and
    # confined, which is what an unclassified command most needs.
    return Segment(Kind.UNKNOWN, _file_operands(argv), head, effect=EFFECT_RUN)


def _git_subcommand(argv: list[str]) -> str:
    """The subcommand in a ``git`` invocation, or "" if there is none to find.

    Global flags come before it and one class of them eats the following token, so a
    scan that stopped at the first non-flag word read ``/repo`` as the subcommand of
    ``git -C /repo status``.
    """
    i = 1
    while i < len(argv):
        tok = argv[i]
        if tok in GIT_VALUE_FLAGS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        return tok
    return ""


def _git_segment(argv: list[str]) -> Segment:
    """Place a ``git`` invocation by what its subcommand does to the repository.

    Read-only subcommands are INSPECT, so plan mode may run ``git status`` and ``git log``;
    everything else stays UNKNOWN (a run).
    """
    sub = _git_subcommand(argv)
    if sub in GIT_READ_SUBCOMMANDS and _git_is_readonly(argv, sub):
        return Segment(Kind.INSPECT, _inspect_operands(argv))
    return Segment(Kind.UNKNOWN, _file_operands(argv), "git", effect=EFFECT_RUN)


def shell_segments(
    command: str, *, allow_expansion: bool = False
) -> list[ParsedSegment] | None:
    """The commands the shell would run, each with its argv and redirection targets, or None.

    None on anything ``shell_paths.parse_segments`` refuses, or a character whose effect
    only the running shell knows. *allow_expansion* lets a plain ``$VAR`` through, for path
    extraction; command substitution stays opaque.
    """
    try:
        segments = parse_segments(command)
    except ShellParseError:
        return None

    if not allow_expansion and unquoted_shell_char(command, _EXPANSION_SIGILS):
        return None
    return segments


def classify_bash_command(
    command: str, *, allow_expansion: bool = False,
) -> list[Segment] | None:
    """Classify *command* into per-segment (kind, operands), or None if opaque.

    An unrecognised command is UNKNOWN (a run), not opaque. A redirection to a file makes a
    read-only segment a write. *allow_expansion* is for ``run_ledger_key`` only.
    """
    parsed = shell_segments(command, allow_expansion=allow_expansion)
    if parsed is None:
        return None

    segments: list[Segment] = []
    for segment in parsed:
        seg = _classify_segment(segment.argv)
        written = segment.write_targets
        if written and seg.kind in READONLY_KINDS:
            seg = Segment(Kind.WRITE, [t for t in written if _looks_like_path(t)])
        segments.append(seg._replace(argv=tuple(segment.argv)))
    return segments


# Command position, for a command the tokenizer could not read: the start, or whatever
# follows a separator. Textual on purpose — the whole point is that shlex already failed.
_OPAQUE_SEGMENT_SPLIT_RE = re.compile(r"[;&|]+")


def opaque_command_executes(command: str) -> bool:
    """Whether a command the classifier gave up on still ran a program.

    Reads command-position tokens only, against ``EXEC_COMMANDS``. Used to demand a verdict,
    never to grant credit.
    """
    for piece in _OPAQUE_SEGMENT_SPLIT_RE.split(command):
        for tok in piece.split():
            if ENV_ASSIGN_RE.match(tok):
                continue
            name = os.path.basename(tok)
            if name in EXEC_COMMANDS or (name and is_path_like_command(tok)):
                return True
            break
    return False


def bash_command_is_readonly(command: str) -> bool:
    """True iff every segment of *command* classifies as a read-only kind.

    Plan mode and the approval exemption rely on it. Anything opaque or unrecognised is not
    read-only; the bash server still validates every call.
    """
    # `srun lscpu` reads nothing it could not read here, but outside an allocation it
    # queues for and spends node hours first: never read-only, never exempt.
    if allocates_cluster(command):
        return False
    segments = classify_bash_command(command)
    if segments is None:
        return False
    return all(seg.kind in READONLY_KINDS for seg in segments)


# Wrappers are unwrapped by the shared ``unwrap_argv`` — the one the bash validator
# uses — so a wrapper's own arguments (``timeout 300``, ``srun -n 4``, ``mpirun -np 8``)
# are never mistaken for the program. ``time`` is handled here: it is a shell keyword
# rather than a command, so it never reaches that validator as a head.
_EXEC_WRAPPERS = COMMAND_WRAPPERS | {"time"}
# Interpreters that execute their first non-flag argument as the real program.
_EXEC_INTERPRETERS = frozenset({
    "python", "python3", "python2", "pypy", "pypy3", "sh", "bash", "zsh",
    "node", "ruby", "perl",
})
# Bounds the peel loop on adversarial input; no real call nests this deep.
_MAX_PEEL = 8


def segment_program(argv: list[str]) -> str | None:
    """The program one segment executes, past assignments and wrappers, or ``None``.

    For an interpreter, its script (``python proxy.py`` runs ``proxy.py``).
    """
    current = list(argv)
    for _ in range(_MAX_PEEL):
        while current and ENV_ASSIGN_RE.match(current[0]) and not current[0].startswith("/"):
            current = current[1:]
        if not current:
            return None
        if os.path.basename(current[0]) == "time":
            current = current[1:]
            while current and current[0].startswith("-"):
                current = current[1:]
            continue
        inner, wrappers = unwrap_argv(current)
        if not wrappers:
            break
        current = inner
    if not current or os.path.basename(current[0]) in _EXEC_WRAPPERS:
        return None
    head = current[0]
    if os.path.basename(head) in _EXEC_INTERPRETERS:
        return next((a for a in current[1:] if not a.startswith("-")), None)
    return head

"""Classify a ``bash_run`` command line into capability *kinds* + operands.

The dual-use bash tool hides its real effect inside a ``command`` string, so the
observation layer (which populates the policy/nudge blackboard by capability) and
the plan-mode/approval gate cannot see whether a call reads, searches, inspects,
writes, or executes. ``classify_bash_command`` closes that gap: it splits the command
into the commands the shell would run (via ``shell_paths.parse_segments``, shared with
the server's validator) and maps each segment's leading command, its options and its
redirections to a :data:`Kind` and the operands it acts on (files read, patterns
searched, dirs inspected, files written…).

Consumers:
- ``bash_command_is_readonly`` (below) — a command is read-only/side-effect-free
  iff every segment is a read-only/discovery kind (see :data:`READONLY_KINDS`); used
  by plan mode and by the all-mode approval exemption.
- ``observations._observe_command`` — credits the blackboard per segment kind.

Conservative by design: any parse ambiguity, a substitution, or an unrecognised
leading command yields ``None`` (opaque → not classifiable), and an ambiguous operand
is dropped rather than guessed. A redirection is not ambiguous, so it is classified
rather than refused: ``> file`` makes the segment a write, an fd redirection changes
nothing. This is NOT a security boundary — the bash server independently validates and
blocks every call; this only observes/derives intent.
"""

from __future__ import annotations

import os
import re
from typing import NamedTuple

# Tokenization, command groups and path-operand extraction all come from the module
# the bash *server* validates with, so classification tracks what the server
# actually permits and confines — one implementation, no drift (see its docstring).
from ....servers._shared.shell_paths import (
    ENV_MANAGER_COMMANDS,
    EXEC_COMMANDS,
    INSPECT_COMMANDS,
    NEUTRAL_COMMANDS,
    READ_COMMANDS,
    SEARCH_COMMANDS,
    WRITE_COMMANDS,
    ParsedSegment,
    ShellParseError,
    is_path_like_command,
    parse_segments,
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


# A token still carrying one of these after tokenization is opaque to *classification*:
# a subshell or substitution runs code, and a ``$VAR``'s value is known only to the
# running shell, so the command's kind would be a guess. (Operator tokens never reach
# here — the shared parser has already lifted redirections out and refused subshells.)
_BASH_FORBIDDEN_CHARS = frozenset("()<>`$&")

# Leading command → kind comes from the shell_paths command *groups*, the same ones the
# bash server builds its allowlist from, so classification cannot drift from what runs.
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

    Only the destination is credited — for `mv a.py b.py` the source no longer
    exists, so crediting it would leave the observation layer tracking a file that
    is gone. `mv`/`cp` put the destination last; `mkdir` creates every positional.
    Best-effort like the rest of this module: a miss (e.g. `cp -t DIR src`) only
    mislabels which file is pending validation, never grants access.
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
    """The path-position arguments of *argv*, as raw tokens.

    Delegates to ``shell_paths.path_operand_tokens`` — the extraction the server
    *confines* — so a file credited here is a file the server judged, and the
    command-family rules (a ``grep`` pattern and a ``sed`` script are data, not
    files) live in one place. Only the extra filter is local: a token that cannot be
    a path (an option, a numeric flag value) is dropped rather than credited.
    """
    return [t for t in path_operand_tokens(argv) if _looks_like_path(t)]


def _read_operands(argv: list[str]) -> list[str]:
    """Path positionals a read command acts on (stdin-only readers have none)."""
    if argv[0] in _STDIN_ONLY_READ:
        return []
    return _file_operands(argv)


def _search_patterns(argv: list[str]) -> list[str]:
    """The pattern(s) a grep/rg segment searches for.

    The pattern is the load-bearing signal of a SEARCH segment (the files it reads
    are incidental and often absent), so it is what the segment carries. Given via
    ``-e``, else the first positional — which is exactly the token
    ``path_operand_tokens`` drops for these commands, the two rules being two sides
    of "the first bare argument of a search is text, not a file".
    """
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
_FILE_LIKE_RE = re.compile(r"\.[A-Za-z0-9]+$")


def _exec_head(argv: list[str]) -> str:
    """The validator-identifying leading token of an exec command.

    ``python -m pytest`` / ``python -m ruff`` report the *module* (``pytest``/``ruff``)
    rather than ``python`` so the observation layer can recognise a validator invoked
    either way; ``python foo.py`` (a script, not ``-m``) and every other command report
    their bare leading command.
    """
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
    """File positionals an exec/validation command acts on (best-effort).

    Only tokens that clearly name a file (``/`` or a dotted extension) are credited.
    Lets the observation layer mark a written file *validated* when the model runs a
    validation command on it via bash — ``python -m py_compile foo.py``,
    ``pytest test_foo.py``, ``ruff check foo.py``, ``mypy foo.py`` all credit their
    file operand, while the module/sub-command tokens are ignored.
    """
    return [
        a for a in argv[1:]
        if not a.startswith("-") and ("/" in a or _FILE_LIKE_RE.search(a))
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


def _classify_segment(argv: list[str]) -> Segment | None:
    """Map one command's argv to a Segment, or None if the leading command is opaque."""
    head = argv[0]

    # A workspace-local executable (./a.out) is an execution. Same test the server
    # uses to decide an argv0 names a program by path rather than by name.
    if is_path_like_command(head):
        return Segment(Kind.EXEC, _exec_operands(argv), head)

    if head == "cd":
        # Target dir (first positional; `cd`, `cd -`, `cd ~` carry none we can rebase).
        target = next((a for a in argv[1:] if not a.startswith("-") and a != "~"), None)
        return Segment(Kind.CHDIR, [target] if target else [])

    if head == "module":
        return _module_segment(argv)
    if head in ENV_MANAGER_COMMANDS:
        return _env_manager_segment(argv)
    if head in EXEC_COMMANDS:
        return Segment(Kind.EXEC, _exec_operands(argv), _exec_head(argv))
    if head in WRITE_COMMANDS:
        return Segment(Kind.WRITE, _destination_operands(argv))
    if head in SEARCH_COMMANDS:
        return Segment(Kind.SEARCH, _search_patterns(argv))
    if head in INSPECT_COMMANDS:
        return Segment(Kind.INSPECT, _inspect_operands(argv))
    if head in READ_COMMANDS:
        if _has_write_flag(argv):
            return Segment(Kind.WRITE, _write_operand(argv))
        return Segment(Kind.READ, _read_operands(argv))
    if head in NEUTRAL_COMMANDS:
        return Segment(Kind.NEUTRAL, [])
    return None  # unrecognised leading command → opaque


def shell_segments(
    command: str, *, allow_expansion: bool = False
) -> list[ParsedSegment] | None:
    """Segment *command* into the commands the shell would run, or None if opaque.

    A thin policy layer over :func:`shell_paths.parse_segments`, which does the
    tokenization the *server's* validator also runs on — so what is one command
    here is one command there, and a redirection target seen here is the file the
    server will confine. Each returned segment carries its argv plus the files its
    redirections touch (``> out.txt``, ``< in.txt``); fd dups and ``/dev/null``
    named no file and are already gone, so the capability probe
    ``which pdflatex 2>/dev/null`` keeps the kind of ``which``.

    Exposed separately from :func:`classify_bash_command` because the
    out-of-workspace gate needs the raw arguments (to resolve the paths they name)
    rather than a kind.

    Returns None on anything the shared parser refuses (substitution, subshell,
    backgrounding, heredoc, quoting error, dangling separator) plus — the part that
    is this side's own conservatism — any token carrying a character whose value
    only the running shell knows.

    *allow_expansion* keeps going through a plain ``$VAR``. A bare variable makes a
    command opaque to **classification** (its kind depends on what the value turns
    out to be), but not to **path extraction**: in flag position it is irrelevant
    (``gcc -I$CUDA_HOME/include a.c``) and in path position it is refused outright
    by the sandbox guard. Without this, such a command yielded no targets at all, so
    a genuine out-of-workspace path sitting beside the flag was never offered to the
    user — refused by the server with no way to grant it. Command *substitution*
    (``$(...)``, ``${...}``, backticks) stays opaque either way: it runs code.
    """
    try:
        segments = parse_segments(command)
    except ShellParseError:
        return None

    forbidden = _BASH_FORBIDDEN_CHARS - {"$"} if allow_expansion else _BASH_FORBIDDEN_CHARS
    for segment in segments:
        for tok in [*segment.argv, *segment.redirect_targets]:
            if any(ch in forbidden for ch in tok):
                return None
            if allow_expansion and ("$(" in tok or "${" in tok):
                return None  # substitution runs code; a bare $VAR does not
    return segments


def classify_bash_command(command: str) -> list[Segment] | None:
    """Classify *command* into per-segment (kind, operands), or None if opaque.

    Opaque for everything :func:`shell_segments` rejects, plus an unrecognised
    leading command in any segment. A returned list always has at least one segment.

    A redirection to a file is a write the leading command's own kind does not
    show: ``ls > out.txt`` creates a file, so the segment cannot stay INSPECT and be
    auto-approved as read-only. A segment whose command is already a write or an
    execution keeps its kind and operands — that command's operands are what the
    observation layer credits (a validator run must still credit the file it
    checked, not the log it was piped into).
    """
    parsed = shell_segments(command)
    if parsed is None:
        return None

    segments: list[Segment] = []
    for segment in parsed:
        seg = _classify_segment(segment.argv)
        if seg is None:
            return None
        written = segment.write_targets
        if written and seg.kind in READONLY_KINDS:
            seg = Segment(Kind.WRITE, [t for t in written if _looks_like_path(t)])
        segments.append(seg)
    return segments


# Command position, for a command the tokenizer could not read: the start, or whatever
# follows a separator. Textual on purpose — the whole point is that shlex already failed.
_OPAQUE_SEGMENT_SPLIT_RE = re.compile(r"[;&|]+")
_OPAQUE_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_]\w*=")


def opaque_command_executes(command: str) -> bool:
    """Whether a command :func:`classify_bash_command` gave up on still *ran* something.

    Classification is all-or-nothing: an inline payload (``python -c "print(f(x))"``),
    a heredoc or a substitution makes the whole command unreadable. That is the right
    answer for *crediting* anything — nothing can be attributed to a command nobody can
    parse — but the wrong one for noticing that a program ran. The head stays legible
    when the rest does not, and the base prompt actively steers one-off checks inline
    (``## Running code``), so the most encouraged idiom was also the least visible one:
    a single pair of parentheses is enough to make ``python -c`` opaque.

    Only command-position tokens are read (start of the command, or after ``; && || |``),
    against the same ``EXEC_COMMANDS`` vocabulary the classifier itself uses, so an
    unparseable ``cat`` or ``grep`` still reports nothing. Deliberately used *only* to
    demand a verdict, never to grant credit.
    """
    for piece in _OPAQUE_SEGMENT_SPLIT_RE.split(command):
        for tok in piece.split():
            if _OPAQUE_ENV_ASSIGN_RE.match(tok):
                continue
            if os.path.basename(tok) in EXEC_COMMANDS or is_path_like_command(tok):
                return True
            break
    return False


def bash_command_is_readonly(command: str) -> bool:
    """True iff *command* is a read-only / side-effect-free shell invocation.

    A command qualifies when it classifies cleanly and *every* segment is a
    read-only/discovery kind (see :data:`READONLY_KINDS`) — so build/execution
    (``gcc``/``python``/``make``/``./a.out``, ``module load``) and in-place writes
    (``sed -i``) do not. Two mode-agnostic consumers rely on it:
    - plan mode, to keep a drafted plan side-effect-free (exec is rejected); and
    - the approval layer, to waive the prompt for read-only bash in **any** mode.

    Conservative by design — on any parse ambiguity or unrecognised leading command
    the classifier returns None and this returns ``False`` (the safe default). The
    bash server still fully validates every accepted call, so this is a policy
    predicate, not the security boundary.
    """
    segments = classify_bash_command(command)
    if segments is None:
        return False
    return all(seg.kind in READONLY_KINDS for seg in segments)

"""The mandatory check, performed here rather than by a tool on the machine.

The check every modified file owes before the run may conclude used to be an external
binary named in the prompt (``ruff``, ``mypy``, ``py_compile``, a compiler). That made
the one blocking axis of the loop depend on what happened to be installed — a missing
linter silently turned a file "unverifiable" — and on the handful of languages those
binaries covered; anything else (``.js``, ``.rs``, ``.go``, ``.sh``) was never checked at
all. So the floor moved in here: it needs nothing but the standard library, it reads
every text file, and it answers the same on every machine.

External checkers did not go away — a ``ruff check`` the model runs still credits the
file and *raises* its tier (``structural`` → ``static``). They are a bonus now, never a
requirement, and nothing in the prompts names one.

Two storeys, both keyed by extension so adding a language is adding a line:

- a **stdlib parser** where one exists (Python, JSON, TOML): the real thing, tier
  ``syntax``, with the offending line in the message;
- a **structural scan** for everything else, tier ``structural``: unbalanced delimiters,
  an unterminated block comment, a leftover conflict marker. It is deliberately narrow —
  it catches the file written half-way, which is the failure the floor exists for, and
  claims nothing about semantics.

Doctrine, everywhere below: **when tokenization is ambiguous, pass.** A false positive
here charges a repair budget against code that is fine; a false negative only leaves us
where we already were.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

# Anything larger is not read: the floor is a cheap check, and a multi-megabyte source
# file is not what it was built for.
_MAX_BYTES = 5 * 1024 * 1024

STATUS_OK = "ok"
STATUS_FAIL = "fail"
STATUS_UNREADABLE = "unreadable"

TIER_STRUCTURAL = "structural"
TIER_SYNTAX = "syntax"


@dataclass(frozen=True)
class CheckOutcome:
    """What the built-in floor found in one file.

    ``status`` is the verdict, ``tier`` what strength of evidence a pass established
    (empty when nothing was established), ``checker`` which storey answered, and
    ``detail`` the one-line diagnostic the model is shown when it failed.
    """

    status: str
    tier: str = ""
    checker: str = ""
    detail: str = ""


# ── The structural scan ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Profile:
    """How to tell code from comment from string in one family of languages.

    Every field exists to *suppress* a false positive rather than to catch anything:
    without the string and comment rules a brace in a message would unbalance the file,
    and ``delimiters=False`` is how a family whose grammar legitimately looks unbalanced
    (shell ``case`` patterns, prose) opts out of that check entirely.
    """

    line_comments: tuple[str, ...] = ()
    block_comment: tuple[str, str] | None = None
    strings: tuple[str, ...] = ()
    backslash_escapes: bool = True
    doubled_quote_escape: bool = False
    delimiters: str = ""            # which opener/closer pairs to balance, "" = none
    skip_preproc: bool = False      # ignore #if/#else/#endif lines when balancing
    keyword_pairs: tuple[tuple[str, str], ...] = ()
    regex_literals: bool = False    # `/.../` is a literal, and its brackets are not code
    heredocs: bool = False          # `<<WORD` swallows the lines up to WORD


_C_FAMILY = _Profile(
    line_comments=("//",),
    block_comment=("/*", "*/"),
    strings=('"', "'", "`"),
    delimiters="()[]{}",
    skip_preproc=True,
)
# ECMAScript and its descendants: same grammar, plus a regular-expression literal whose
# character classes would otherwise read as unbalanced brackets. Six of this repo's own
# files failed on exactly that before the literal was taught to the scan.
_JS_FAMILY = _Profile(
    line_comments=("//",),
    block_comment=("/*", "*/"),
    strings=('"', "'", "`"),
    delimiters="()[]{}",
    regex_literals=True,
)
_FORTRAN = _Profile(
    line_comments=("!",),
    strings=('"', "'"),
    backslash_escapes=False,
    doubled_quote_escape=True,
    delimiters="()[]",
)
_SHELL = _Profile(
    line_comments=("#",),
    strings=('"', "'"),
    # `case x in  pattern)` closes a paren nothing opened, and `${a[@]}` nests brackets
    # inside expansions: delimiter balance says nothing true about a shell script.
    # What truncation *does* break is the block keywords, so those are counted instead.
    keyword_pairs=(("if", "fi"), ("do", "done"), ("case", "esac")),
    heredocs=True,
)
# Prose, and the default for an unknown extension: no grammar is assumed, so only the
# things that are wrong in any text file are looked for.
_PROSE = _Profile()

_PROFILE_BY_EXTENSION: dict[str, _Profile] = {
    **{
        ext: _C_FAMILY
        for ext in (
            ".c", ".h", ".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".h++",
            ".cu", ".cuh", ".java",
            ".rs", ".go", ".cs", ".swift", ".kt", ".kts", ".scala", ".php", ".dart",
            ".glsl", ".cl", ".proto", ".css", ".scss",
        )
    },
    **{
        ext: _JS_FAMILY
        for ext in (".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx")
    },
    **{
        ext: _FORTRAN
        for ext in (
            ".f", ".for", ".ftn", ".f77", ".f90", ".f95", ".f03", ".f08", ".f18",
        )
    },
    **{ext: _SHELL for ext in (".sh", ".bash", ".zsh", ".ksh")},
    **{
        ext: _PROSE
        for ext in (".md", ".rst", ".txt", ".tex", ".csv", ".org", ".adoc")
    },
}

_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_CONFLICT_MARKERS = ("<<<<<<< ", ">>>>>>> ")
_PREPROC_RE = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_HEREDOC_RE = re.compile(r"<<-?\s*([\"\']?)([A-Za-z_][A-Za-z0-9_]*)\1")
# After these, a `/` opens a regular expression; after an identifier or a closing
# bracket it is division. The standard disambiguation, and the reason it is a list of
# keywords rather than a rule: `return /x/` and `a / b` differ only by what precedes.
_REGEX_KEYWORDS = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void",
    "do", "else", "case", "yield", "await", "throw",
})


def _structural_scan(text: str, profile: _Profile) -> str:
    """Return a one-line diagnostic, or "" when nothing structural is wrong."""
    for line_no, line in enumerate(text.splitlines(), 1):
        if line.startswith(_CONFLICT_MARKERS):
            return f"line {line_no}: leftover merge conflict marker"

    stack: list[tuple[str, int]] = []
    line_no = 1
    i = 0
    n = len(text)
    in_block_comment_at = 0
    block_open, block_close = profile.block_comment or ("", "")
    balancing = bool(profile.delimiters)
    skip_line = False
    at_line_start = True
    # Whether a `/` here would open a regular expression rather than divide. Only the
    # ECMAScript profile reads it; it is maintained by every branch that consumes a
    # token, which is why those branches set it even where nothing looks at it.
    regex_allowed = True

    while i < n:
        ch = text[i]
        if ch == "\n":
            line_no += 1
            i += 1
            skip_line = False
            at_line_start = True
            continue
        if at_line_start:
            at_line_start = False
            if profile.skip_preproc and _PREPROC_RE.match(text, i):
                skip_line = True
        if skip_line or ch in " \t":
            i += 1
            continue

        if block_open and text.startswith(block_open, i):
            in_block_comment_at = line_no
            end = text.find(block_close, i + len(block_open))
            if end < 0:
                return f"line {in_block_comment_at}: block comment is never closed"
            line_no += text.count("\n", i, end)
            i = end + len(block_close)
            continue

        if any(text.startswith(tok, i) for tok in profile.line_comments):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue

        if profile.heredocs and text.startswith("<<", i):
            match = _HEREDOC_RE.match(text, i)
            if match:
                i, line_no = _skip_heredoc(text, match.end(), line_no, match.group(2))
                continue

        if profile.regex_literals and ch == "/" and regex_allowed:
            end = _skip_regex(text, i)
            if end > 0:
                i = end
                regex_allowed = False
                continue

        if ch in profile.strings:
            regex_allowed = False
            i = _skip_string(text, i, profile)
            # A string that reaches the end of its line is closed there rather than
            # reported: heredocs, template literals and continuations all span lines
            # legitimately, and guessing which is which is how a scan invents defects.
            continue

        if balancing and ch in profile.delimiters:
            regex_allowed = ch not in ")]"
            if ch in _OPEN_TO_CLOSE:
                stack.append((ch, line_no))
            else:
                if not stack:
                    return f"line {line_no}: closing '{ch}' with nothing open"
                opener, opened_at = stack.pop()
                if _OPEN_TO_CLOSE[opener] != ch:
                    return (
                        f"line {line_no}: '{ch}' closes the '{opener}' opened on "
                        f"line {opened_at}"
                    )
            i += 1
            continue

        if (profile.keyword_pairs or profile.regex_literals) and (ch.isalpha() or ch == "_"):
            match = _WORD_RE.match(text, i)
            if match:
                word = match.group()
                regex_allowed = word in _REGEX_KEYWORDS
                if profile.keyword_pairs:
                    _count_keyword(word, line_no, profile, stack)
                i = match.end()
                continue

        regex_allowed = not ch.isalnum()
        i += 1

    if stack:
        opener, opened_at = stack[0]
        return f"line {opened_at}: '{opener}' is never closed (file ends inside it)"
    return ""


def _skip_string(text: str, start: int, profile: _Profile) -> int:
    """Index just past the string literal opening at *start*, or past its line."""
    quote = text[start]
    i = start + 1
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            return i          # unterminated on this line: give up, do not accuse
        if profile.backslash_escapes and ch == "\\":
            i += 2
            continue
        if ch == quote:
            if profile.doubled_quote_escape and i + 1 < n and text[i + 1] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def _skip_heredoc(text: str, start: int, line_no: int, word: str) -> tuple[int, int]:
    """Index and line just past a ``<<WORD`` heredoc body, which is data, not code.

    Without this, the script embedding a Python snippet gets its ``if`` counted and is
    reported as never closed — the shape ``install.sh`` has.
    """
    end_of_line = text.find("\n", start)
    if end_of_line < 0:
        return len(text), line_no
    i = end_of_line + 1
    line_no += 1
    n = len(text)
    while i < n:
        stop = text.find("\n", i)
        line = text[i:n if stop < 0 else stop]
        if line.strip() == word:
            return (n, line_no) if stop < 0 else (stop, line_no)
        if stop < 0:
            return n, line_no
        i = stop + 1
        line_no += 1
    return n, line_no


def _skip_regex(text: str, start: int) -> int:
    """Index just past the ``/.../flags`` literal at *start*, or 0 if it is not one.

    A literal never spans a line, so an unterminated one means the `/` was division
    after all — which is the conservative reading anyway.
    """
    i = start + 1
    n = len(text)
    in_class = False
    while i < n:
        ch = text[i]
        if ch == "\n":
            return 0
        if ch == "\\":
            i += 2
            continue
        if ch == "[":
            in_class = True
        elif ch == "]":
            in_class = False
        elif ch == "/" and not in_class:
            i += 1
            while i < n and text[i].isalpha():
                i += 1
            return i
        i += 1
    return 0


def _count_keyword(
    word: str, line_no: int, profile: _Profile, stack: list[tuple[str, int]],
) -> None:
    """Push/pop the block-keyword stack shared with the delimiter one.

    Shell has no delimiter balance worth trusting, so the stack is free for its block
    words; a language never uses both.
    """
    for opener, closer in profile.keyword_pairs:
        if word == opener:
            stack.append((opener, line_no))
            return
        if word == closer:
            for index in range(len(stack) - 1, -1, -1):
                if stack[index][0] == opener:
                    del stack[index]
                    return
            return      # a closer with nothing open: `esac` in a string we mis-read


# ── The stdlib parsers ────────────────────────────────────────────────────────


def _check_python(text: str, path: str) -> str | None:
    try:
        compile(text, path, "exec")
    except SyntaxError as exc:
        where = f"line {exc.lineno}: " if exc.lineno else ""
        return f"{where}{exc.msg}"
    except (ValueError, MemoryError, RecursionError) as exc:
        return f"cannot be parsed: {exc}"
    return ""


def _check_json(text: str, path: str) -> str | None:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        return f"line {exc.lineno}: {exc.msg}"
    except (ValueError, RecursionError) as exc:
        return f"cannot be parsed: {exc}"
    return ""


def _check_toml(text: str, path: str) -> str | None:
    try:
        import tomllib
    except ImportError:
        return None     # older interpreter: hand the file to the structural scan
    try:
        tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return f"cannot be parsed: {exc}"
    except (ValueError, RecursionError) as exc:
        return f"cannot be parsed: {exc}"
    return ""


# Entity declarations are the billion-laughs vector, and the stdlib parser expands
# internal entities. Rather than carry a defusing parser we hand such a file to the
# structural scan: declining to answer is always available, and the floor's whole
# doctrine is that an uncertain reading passes.
_XML_ENTITY_RE = re.compile(r"<!ENTITY\b", re.IGNORECASE)
# A section header is what makes a file INI at all. `.cfg` is claimed by tools that put
# something else entirely in it, and configparser's first complaint about those is
# "missing section header" — a defect report about the wrong grammar. So a file with no
# header is not INI, and the structural scan takes it.
_INI_SECTION_RE = re.compile(r"^[ \t]*\[[^]\n]+\][ \t]*$", re.MULTILINE)


def _check_xml(text: str, path: str) -> str | None:
    if _XML_ENTITY_RE.search(text):
        return None
    from xml.etree import ElementTree
    try:
        ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        line = getattr(exc, "position", (0, 0))[0]
        return f"line {line}: {exc.msg}" if line else str(exc)
    except (ValueError, RecursionError) as exc:
        return f"cannot be parsed: {exc}"
    return ""


def _check_ini(text: str, path: str) -> str | None:
    if not _INI_SECTION_RE.search(text):
        return None
    import configparser
    # Raw: interpolation would raise on a legitimate `%(name)s` whose key lives in a
    # section this parser never reads, which is a defect report about nothing.
    parser = configparser.RawConfigParser(strict=True)
    try:
        parser.read_string(text, source=path)
    except configparser.ParsingError as exc:
        # `.errors` is [(lineno, repr(line)), …]; the message alone carries the source
        # path and no position, which is the wrong half of the two. The offending line
        # is read back from the text so it is quoted as written, not as a repr.
        line = exc.errors[0][0] if exc.errors else 0
        source = text.splitlines()[line - 1].strip() if 0 < line <= len(text.splitlines()) else ""
        return f"line {line}: not a key/value pair or a section header: {source[:60]}"
    except configparser.Error as exc:
        line = getattr(exc, "lineno", 0)
        message = str(exc).split("]: ")[-1].splitlines()[0]
        return f"line {line}: {message}" if line else message
    except (ValueError, RecursionError) as exc:
        return f"cannot be parsed: {exc}"
    return ""


_PARSER_BY_EXTENSION = {
    ".py": _check_python,
    ".pyi": _check_python,
    ".pyw": _check_python,
    ".json": _check_json,
    ".toml": _check_toml,
    # XML and its dialects — real well-formedness rather than a delimiter count. HTML
    # is deliberately absent: it is not well-formed XML and never was.
    **{
        ext: _check_xml
        for ext in (
            ".xml", ".xsd", ".xsl", ".xslt", ".svg", ".plist",
            ".vcxproj", ".csproj", ".ui", ".rng", ".wsdl",
        )
    },
    ".ini": _check_ini,
    ".cfg": _check_ini,
}


def check_file(path: str) -> CheckOutcome:
    """Run the floor over one file.

    A parser answers where the extension has one; everything else gets the structural
    scan. A file that cannot be decoded as text is neither: it is reported unreadable,
    which is the one honest "nothing here could check this" left.
    """
    try:
        if os.path.getsize(path) > _MAX_BYTES:
            return CheckOutcome(STATUS_UNREADABLE, detail="file is too large to check here")
        with open(path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        return CheckOutcome(STATUS_UNREADABLE, detail=f"cannot be read: {exc}")
    if b"\0" in raw:
        return CheckOutcome(STATUS_UNREADABLE, detail="binary content")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return CheckOutcome(STATUS_UNREADABLE, detail="not valid UTF-8")

    extension = os.path.splitext(path)[1].lower()
    parser = _PARSER_BY_EXTENSION.get(extension)
    if parser is not None:
        # ``None`` means the parser could not answer here (a stdlib module this
        # interpreter predates): the structural scan takes the file instead, rather
        # than the environment deciding again what gets checked.
        detail = parser(text, path)
        if detail is not None:
            checker = parser.__name__.removeprefix("_check_")
            if detail:
                return CheckOutcome(STATUS_FAIL, TIER_SYNTAX, checker, detail)
            return CheckOutcome(STATUS_OK, TIER_SYNTAX, checker)

    profile = _PROFILE_BY_EXTENSION.get(extension, _PROSE)
    detail = _structural_scan(text, profile)
    if detail:
        return CheckOutcome(STATUS_FAIL, TIER_STRUCTURAL, "structural", detail)
    return CheckOutcome(STATUS_OK, TIER_STRUCTURAL, "structural")


# ── The sweep ─────────────────────────────────────────────────────────────────


def _stamp(path: str) -> tuple[int, int]:
    try:
        info = os.stat(path)
    except OSError:
        return (0, 0)
    return (info.st_mtime_ns, info.st_size)


def sweep_builtin_checks(execution_context: dict, absolute_path_fn=None) -> None:
    """Check every file that still owes one, once each, on its final content.

    Called where the loop asks whether it may conclude — not after each write. A file
    the model edits five times is checked once, on what it ended up being, and the
    stamp is what makes that true across the several places the gate is consulted:
    an unchanged file is skipped, one edited since is checked again.
    """
    from .observations import _mark_file_validated, _register_validation_failure
    from .workflow import pending_validation_paths

    resolve = absolute_path_fn or os.path.abspath
    stamps = execution_context.setdefault("builtin_check_stamp", {})
    findings = execution_context.setdefault("builtin_check_findings", {})

    for path in pending_validation_paths(execution_context):
        absolute = resolve(path)
        stamp = _stamp(absolute)
        if stamps.get(path) == stamp:
            continue
        stamps[path] = stamp
        outcome = check_file(absolute)
        if outcome.status == STATUS_UNREADABLE:
            findings.pop(path, None)
            execution_context["unverifiable_files"].add(path)
        elif outcome.status == STATUS_FAIL:
            findings[path] = outcome.detail
            _register_validation_failure(execution_context, path, outcome.tier)
        else:
            findings.pop(path, None)
            _mark_file_validated(execution_context, path, outcome.tier)


def builtin_check_failures(execution_context: dict) -> dict[str, str]:
    """Files the floor rejected and that are still dirty, path -> diagnostic."""
    findings = execution_context.get("builtin_check_findings") or {}
    validated = execution_context.get("validated_files", set())
    return {p: d for p, d in sorted(findings.items()) if p not in validated}

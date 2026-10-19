"""Editing text as the file tools do it, without touching the disk.

A line range or an anchored span replaced (with the fallbacks that recover an anchor
copied with line numbers or with stripped trailing spaces), every occurrence found or
replaced, an edit already applied recognised, and what a failed anchor shows instead:
the closest current text, ready to copy back.
"""

import ast
import bisect
import difflib
import re

# Lines shown either side of a near-match when an anchor fails, so the excerpt can be
# copied back into old_text without a further read.
ANCHOR_SNIPPET_MARGIN = 3
_CONTEXT_LINES = 2


def count_lines(text: str) -> int:
    return len(text.splitlines())


def _preferred_newline(existing_text: str) -> str:
    if "\r\n" in existing_text:
        return "\r\n"
    if "\r" in existing_text:
        return "\r"
    return "\n"


def _ensure_final_newline_if_needed(text: str, newline: str) -> list[str]:
    """
    Split into lines preserving line endings; if non-empty and missing a final newline,
    append one using the preferred newline style.
    """
    parts = text.splitlines(True)
    if text and not text.endswith(("\n", "\r")):
        if parts:
            parts[-1] = parts[-1] + newline
        else:
            parts = [newline]
    return parts


def unified_diff(old_lines: list[str], new_lines: list[str], path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=path,
            tofile=path,
            lineterm="\n",
        )
    )


def _changed_line_span(old_lines: list[str], new_lines: list[str]) -> tuple[int, int]:
    """1-based inclusive span of *new_lines* that differs, by common prefix/suffix.

    Derived from the two contents rather than from each tool's own arguments, so every
    edit tool gets the same answer without any of them tracking a match position.
    A pure deletion returns ``end < start`` — nothing new occupies the site.
    """
    n_old, n_new = len(old_lines), len(new_lines)
    pre = 0
    while pre < n_old and pre < n_new and old_lines[pre] == new_lines[pre]:
        pre += 1
    suf = 0
    while (
        suf < n_old - pre
        and suf < n_new - pre
        and old_lines[n_old - 1 - suf] == new_lines[n_new - 1 - suf]
    ):
        suf += 1
    return pre + 1, n_new - suf


def edit_orientation(content: str, updated: str) -> dict:
    """Where an edit landed: the new span and the line shift below it."""
    old_lines = content.splitlines(True)
    new_lines = updated.splitlines(True)
    start, end = _changed_line_span(old_lines, new_lines)
    total = len(new_lines)
    return {
        "new_start_line": start,
        "new_end_line": end,
        "total_lines": total,
        "line_delta": total - len(old_lines),
    }


def validate_python_content(path: str, content: str) -> str | None:
    """Return an error message if the final content of a .py file is syntactically invalid."""
    if not path.lower().endswith(".py"):
        return None
    try:
        ast.parse(content)
        return None
    except SyntaxError as syn_err:
        return (
            f"Syntax error in resulting Python content — file NOT written: {syn_err.msg} "
            f"(line {syn_err.lineno}, col {syn_err.offset})."
        )


def apply_replace_lines_to_text(content: str, start_line: int, end_line: int, new_content: str) -> tuple[str | None, str | None]:
    """
    Replace a strict 1-based inclusive line range in text.
    Returns (updated_text, error_message).
    """
    old_lines = content.splitlines(True)
    total = len(old_lines)

    if total == 0:
        return None, "Cannot replace lines in an empty file."

    if start_line < 1:
        return None, f"start_line must be >= 1 (got {start_line})."
    if end_line < start_line:
        return None, f"end_line must be >= start_line (got {start_line}-{end_line})."
    if start_line > total or end_line > total:
        return None, f"Requested line range {start_line}-{end_line} is outside file bounds (file has {total} lines)."

    newline = _preferred_newline(content)
    new_block = _ensure_final_newline_if_needed(new_content, newline)
    new_lines = old_lines[: start_line - 1] + new_block + old_lines[end_line:]
    return "".join(new_lines), None


def nearest_context(content: str, old_text: str, *, max_snippets: int = 3) -> str:
    """Return a short excerpt showing where *old_text* almost matches.

    Wide enough to be copied straight into ``old_text``: a failed anchor is a request
    for the exact current text, and answering it with the text costs a few dozen tokens
    where answering it with "go read the file" costs a round trip and a re-read.
    """
    probe = ""
    for part in old_text.splitlines():
        stripped = part.strip()
        if stripped:
            probe = stripped[:80]
            break
    if not probe:
        return ""

    lines = content.splitlines()
    hits: list[str] = []
    # Compared with whitespace collapsed: differing spacing is the single most common
    # reason an anchor misses, and it is exactly the case where showing the real text
    # is the whole answer. A literal substring test never matches there.
    probe_norm = " ".join(probe.lower().split())
    if not probe_norm:
        return ""

    for idx, line in enumerate(lines):
        if probe_norm in " ".join(line.lower().split()):
            start = max(0, idx - ANCHOR_SNIPPET_MARGIN)
            end = min(len(lines), idx + ANCHOR_SNIPPET_MARGIN + 1)
            snippet = "\n".join(
                f"  {start + i + 1}: {lines[start + i]}"
                for i in range(end - start)
            )
            hits.append(snippet)
            if len(hits) >= max_snippets:
                break

    if not hits:
        return ""

    return "Closest matches (case-insensitive) for " + repr(probe) + ":\n" + "\n---\n".join(hits)


def strip_line_number_prefixes(text: str, expect_start: int | None = None) -> str | None:
    """*text* without consecutive ``N: `` prefixes (from *expect_start*), or None if not numbered."""
    lines = text.splitlines()
    if not lines:
        return None
    out: list[str] = []
    previous: int | None = None
    for line in lines:
        match = re.match(r"^\s*(\d+): ?", line)
        if not match:
            return None
        number = int(match.group(1))
        if previous is None:
            if expect_start is not None and number != expect_start:
                return None
        elif number != previous + 1:
            return None
        previous = number
        out.append(line[match.end():])
    return "\n".join(out)


def anchor_retry_hint(excerpt: str) -> str:
    """How to retry a failed anchor — from the excerpt when there is one.

    Sending the caller off to read the file is only the right answer when nothing in
    this reply shows the current text; when the excerpt is right there, telling it to
    read anyway is what turns one failed edit into a search and a read.
    """
    if excerpt:
        return (
            "Copy old_text verbatim from anchor_excerpt — including indentation and "
            "trailing whitespace — and drop the line-number prefix ('  12: '). Read the "
            "file only if the excerpt does not cover the site you meant."
        )
    return (
        "Use read_file_lines to inspect the current content and copy the exact "
        "whitespace and text for old_text, without the line-number prefix ('12: ')."
    )


def _detect_context_bleed(
    content: str,
    match_pos: int,
    old_len: int,
    new_text: str,
) -> str | None:
    """An error when *new_text* repeats a line adjacent to the replaced text (it would duplicate)."""
    before = content[:match_pos]
    after = content[match_pos + old_len:]

    # Collect the last non-blank line before the match and first after.
    before_last = next(
        (ln.rstrip() for ln in reversed(before.splitlines()) if ln.strip()),
        None,
    )
    after_first = next(
        (ln.rstrip() for ln in after.splitlines() if ln.strip()),
        None,
    )

    # Collect first and last non-blank lines of new_text.
    new_lines_nonempty = [ln.rstrip() for ln in new_text.splitlines() if ln.strip()]
    if not new_lines_nonempty:
        return None

    new_first = new_lines_nonempty[0]
    new_last = new_lines_nonempty[-1]

    # Minimum length to avoid flagging trivial lines like "pass", "}", "return".
    _MIN_LEN = 8

    if before_last and len(before_last.strip()) >= _MIN_LEN and new_first == before_last:
        return (
            f"new_text starts with a line already present immediately before old_text: "
            f"{before_last.strip()!r}. "
            "new_text must contain ONLY the replacement for old_text — "
            "do not include surrounding context lines that are already in the file. "
            "Remove the leading context line(s) from new_text."
        )

    if after_first and len(after_first.strip()) >= _MIN_LEN and new_last == after_first:
        return (
            f"new_text ends with a line already present immediately after old_text: "
            f"{after_first.strip()!r}. "
            "new_text must contain ONLY the replacement for old_text — "
            "do not include surrounding context lines that are already in the file. "
            "Remove the trailing context line(s) from new_text."
        )

    return None


def apply_replace_in_text(content: str, old_text: str, new_text: str) -> tuple[str | None, str | None, bool]:
    """Replace exactly one occurrence of old_text in content.

    Returns (new_content, error_msg, normalized_match).
    - Success: (new_content, None, False) or (new_content, None, True) if whitespace-normalized.
    - Failure: (None, error_msg, False).
    """
    count = content.count(old_text)
    if count == 0:
        # An anchor copied from numbered output, prefixes and all. Only accepted when
        # the un-numbered form matches the file exactly once, so a line of code that
        # merely looks numbered cannot be rewritten by this path.
        unnumbered = strip_line_number_prefixes(old_text)
        if unnumbered is not None and content.count(unnumbered) == 1:
            return apply_replace_in_text(
                content, unnumbered, strip_line_number_prefixes(new_text) or new_text
            )
        # Trailing-whitespace-stripped fallback: strip trailing spaces/tabs per
        # line and retry the search on the file as a line-by-line window scan.
        # This recovers from anchors the model produced from memory with minor
        # whitespace differences (e.g. trailing spaces the formatter stripped).
        old_lines = old_text.splitlines(keepends=False)
        n = len(old_lines)
        if n > 0:
            stripped_old = "\n".join(line.rstrip() for line in old_lines)
            content_lines = content.splitlines(keepends=True)
            match_start: int | None = None
            match_count = 0
            for i in range(max(0, len(content_lines) - n + 1)):
                window = content_lines[i : i + n]
                if len(window) < n:
                    break
                window_stripped = "\n".join(line.rstrip() for line in window)
                if window_stripped == stripped_old:
                    match_count += 1
                    match_start = i
            if match_count == 1 and match_start is not None:
                # Compute byte position of fuzzy match for context-bleed check.
                fuzzy_match_pos = len("".join(content_lines[:match_start]))
                fuzzy_old_len = len("".join(content_lines[match_start : match_start + n]))
                bleed_err = _detect_context_bleed(content, fuzzy_match_pos, fuzzy_old_len, new_text)
                if bleed_err:
                    return None, bleed_err, False
                new_lines = new_text.splitlines(keepends=True)
                # Preserve trailing newline style from the replaced block.
                last_orig = content_lines[match_start + n - 1]
                if new_lines and not new_lines[-1].endswith(("\n", "\r")):
                    newline = "\r\n" if last_orig.endswith("\r\n") else "\n"
                    new_lines[-1] += newline
                replaced = "".join(
                    content_lines[:match_start] + new_lines + content_lines[match_start + n :]
                )
                return replaced, None, True
        ctx = nearest_context(content, old_text)
        msg = "Target text was not found."
        if ctx:
            msg += " Closest matches are in anchor_excerpt."
        return None, msg, False
    if count > 1:
        return None, "Target text appears multiple times.", False
    match_pos = content.index(old_text)
    bleed_err = _detect_context_bleed(content, match_pos, len(old_text), new_text)
    if bleed_err:
        return None, bleed_err, False
    return content.replace(old_text, new_text, 1), None, False


def edit_already_applied(content: str, old_text: str, new_text: str) -> bool:
    """True when the anchor is gone and the edit's result (a deletion, or *new_text* of 4+
    characters) is already in the file.
    """
    if not old_text.strip() or old_text in content:
        return False
    if not new_text.strip():
        return True  # deletion whose target is already gone
    return new_text in content and len(new_text.strip()) >= 4


def already_applied_result(path: str, fp: str) -> dict:
    """Uniform no-op payload for an edit whose change is already present."""
    return {
        "operation": "noop",
        "path": path,
        "absolute_path": fp,
        "reason": "No change: the original text is absent and the intended result is "
                  "already present — this edit was already applied.",
    }


def _build_replace_all_pattern(old_text: str, *, whole_word: bool) -> re.Pattern[str]:
    """The regex replacing *old_text* everywhere; *whole_word* adds a boundary only at identifier ends."""
    escaped = re.escape(old_text)
    if not whole_word:
        return re.compile(escaped)

    starts_with_word = old_text[0].isalnum() or old_text[0] == "_"
    ends_with_word = old_text[-1].isalnum() or old_text[-1] == "_"

    prefix = r"(?<![A-Za-z0-9_])" if starts_with_word else ""
    suffix = r"(?![A-Za-z0-9_])" if ends_with_word else ""
    return re.compile(prefix + escaped + suffix)


def _line_starts(content: str) -> list[int]:
    starts = [0]
    for idx, ch in enumerate(content):
        if ch == "\n":
            starts.append(idx + 1)
    return starts


def find_all_occurrences(content: str, old_text: str, *, whole_word: bool) -> list[dict]:
    """Every occurrence of *old_text* (multi-line too): ``[{"line", "column", "context"}]``."""
    pattern = _build_replace_all_pattern(old_text, whole_word=whole_word)
    matches = list(pattern.finditer(content))

    if not matches:
        return []

    lines = content.splitlines(True)
    starts = _line_starts(content)
    out: list[dict] = []

    for m in matches:
        pos = m.start()
        line_no = bisect.bisect_right(starts, pos)
        line_start = starts[line_no - 1]
        column = pos - line_start

        ctx_start_line = max(1, line_no - _CONTEXT_LINES)
        ctx_end_line = min(len(lines), line_no + _CONTEXT_LINES)
        context = "".join(lines[ctx_start_line - 1: ctx_end_line])

        out.append({
            "line": line_no,
            "column": column,
            "context": context,
        })

    return out


def replace_all_in_text(content: str, old_text: str, new_text: str, *, whole_word: bool) -> tuple[str, int]:
    pattern = _build_replace_all_pattern(old_text, whole_word=whole_word)
    updated, count = pattern.subn(new_text, content)
    return updated, count

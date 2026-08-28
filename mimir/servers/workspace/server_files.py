import ast
import bisect
import difflib
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok
from root_paths import require_absolute, resolve_path_in_root
from approved_roots import approved_roots
from state_paths import standing_roots
from capabilities import (
    tool_caps,
    CONTENT_WRITE,
    EDIT,
    OVERWRITE,
    REMOVE,
    REPLACEMENT_TRACK,
    RECOVERABLE,
)

ROOT_DIR_ABS = os.path.abspath(
    os.environ.get("MCP_FILES_ROOT", os.getcwd())
)

mcp = FastMCP(
    "FileServer",
    debug=False,
    log_level="ERROR",
)

_MIN_REPLACE_ALL_PATTERN_LEN = 4
# Lines shown either side of a near-match when an anchor fails, so the excerpt can be
# copied back into old_text without a further read.
_ANCHOR_SNIPPET_MARGIN = 3
_CONTEXT_LINES = 2


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe(path: str) -> str:
    """Resolve absolute path from input (absolute path or path relative to ROOT_DIR_ABS).

    Out-of-workspace paths the user approved this session (read/write/run) are read
    per-call from the shared allowlist so an approved edit outside the workspace is
    honored without respawning the server. The scratchpad is granted alongside them
    as a standing system root — writable without approval, since nothing there is a
    deliverable (see state_paths.scratch_dir).
    """
    return resolve_path_in_root(path, ROOT_DIR_ABS, "file root",
                                extra_roots=list(approved_roots()) + standing_roots())


def _require_abs(path: str, arg: str = "path") -> dict | None:
    """Reject a relative *path* for a file tool (see root_paths.require_absolute)."""
    return require_absolute(path, ROOT_DIR_ABS, arg)


def _read_text(fp: str) -> str:
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_lines(fp: str) -> list[str]:
    """Read file as a list of lines (preserving line endings)."""
    with open(fp, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()


def _count_lines(text: str) -> int:
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


def _unified_diff(old_lines: list[str], new_lines: list[str], path: str) -> str:
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


def _edit_orientation(content: str, updated: str) -> dict:
    """Where the edit landed in the file it just produced.

    An edit renumbers every line below it, so the line numbers the caller used to find
    the site — from a search, an outline, an earlier read — stop describing the file the
    moment the write succeeds. Handing back the new span and the shift applied below it
    is what lets the next edit be aimed without first re-reading the file; the text
    itself is already in the returned diff.
    """
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


def _validate_python_content(path: str, content: str) -> str | None:
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


def _atomic_write_text(fp: str, content: str) -> None:
    """Atomically replace a file's content."""
    dir_name = os.path.dirname(fp) or "."
    os.makedirs(dir_name, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".mcp_tmp_", suffix=".tmp", dir=dir_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as tmp:
            tmp.write(content)
        os.replace(tmp_path, fp)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        finally:
            raise


def _write_validated_text(fp: str, path: str, content: str) -> None:
    syntax_error = _validate_python_content(path, content)
    if syntax_error:
        raise ValueError(syntax_error)
    _atomic_write_text(fp, content)


def _apply_replace_lines_to_text(content: str, start_line: int, end_line: int, new_content: str) -> tuple[str | None, str | None]:
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


def _nearest_context(content: str, old_text: str, *, max_snippets: int = 3) -> str:
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
            start = max(0, idx - _ANCHOR_SNIPPET_MARGIN)
            end = min(len(lines), idx + _ANCHOR_SNIPPET_MARGIN + 1)
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


def _strip_line_number_prefixes(text: str, expect_start: int | None = None) -> str | None:
    """*text* without its ``N: `` line-number prefixes, or ``None`` if it has none.

    Reads, search excerpts and the post-edit window all number the text they hand back,
    so the text a model has in front of it is numbered and the anchor it is asked for is
    not. Undoing the prefix here costs a few lines; making the model do it costs a
    failed edit and the re-read that follows.

    Numbering must run consecutively — the text was copied from one contiguous block or
    it was not — and *expect_start* pins where it must begin. Real code that happens to
    look numbered (``1: value``) does not survive either test, and callers verify what
    comes back besides.
    """
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


def _anchor_retry_hint(excerpt: str) -> str:
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
    """Return an error message when new_text includes lines already adjacent to old_text.

    This is the most common cause of duplicated code after replace_in_file: the model
    provides new_text that contains context lines from before or after old_text, so those
    lines end up appearing twice in the resulting file.

    Only flags clear cases where a non-trivial line at the start/end of new_text exactly
    matches the immediately adjacent line in the file.
    """
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


def _apply_replace_in_text(content: str, old_text: str, new_text: str) -> tuple[str | None, str | None, bool]:
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
        unnumbered = _strip_line_number_prefixes(old_text)
        if unnumbered is not None and content.count(unnumbered) == 1:
            return _apply_replace_in_text(
                content, unnumbered, _strip_line_number_prefixes(new_text) or new_text
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
        ctx = _nearest_context(content, old_text)
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


def _edit_already_applied(content: str, old_text: str, new_text: str) -> bool:
    """True when a text edit's intended result is already present and its anchor gone.

    Detects a benign re-issue of an edit a prior call already made: the ``old_text``
    anchor is absent AND either the edit was a deletion (empty ``new_text``) or the
    ``new_text`` is already in the file. Callers use this to report a clear no-op
    instead of a "not found" error that reads as a failure while the file already
    holds the change. Requires a non-trivial ``new_text`` (>= 4 non-space chars) so a
    genuinely wrong anchor with a tiny replacement is not silently swallowed.
    """
    if not old_text.strip() or old_text in content:
        return False
    if not new_text.strip():
        return True  # deletion whose target is already gone
    return new_text in content and len(new_text.strip()) >= 4


def _already_applied_result(path: str, fp: str) -> dict:
    """Uniform no-op payload for an edit whose change is already present."""
    return {
        "operation": "noop",
        "path": path,
        "absolute_path": fp,
        "reason": "No change: the original text is absent and the intended result is "
                  "already present — this edit was already applied.",
    }


def _build_replace_all_pattern(old_text: str, *, whole_word: bool) -> re.Pattern[str]:
    """
    Build a regex pattern for replacing old_text everywhere.

    When whole_word=True:
    - add a left boundary only if old_text starts with an identifier char
    - add a right boundary only if old_text ends with an identifier char

    This lets patterns like "math." match in "math.sin(...)" while still
    preventing "math" from matching inside "aftermath".
    """
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


def _find_all_occurrences(content: str, old_text: str, *, whole_word: bool) -> list[dict]:
    """
    Find all occurrences of old_text in content, including multi-line matches.

    Returns:
        [
            {
                "line": <1-based line>,
                "column": <0-based column>,
                "context": <nearby text>
            },
            ...
        ]
    """
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


def _replace_all_in_text(content: str, old_text: str, new_text: str, *, whole_word: bool) -> tuple[str, int]:
    pattern = _build_replace_all_pattern(old_text, whole_word=whole_word)
    updated, count = pattern.subn(new_text, content)
    return updated, count


# ── tools ─────────────────────────────────────────────────────────────────────

# Not exposed as a tool — directory listing is owned by the search server's
# list_directory. Retained as a plain helper backing the files://list resource.
def list_files(subdir: str = ".") -> dict:
    """List all files and sub-directories inside a directory.

    Returns {"status": "ok", "entries": [{"name", "type", "size_kb"}], "count": <n>}.

    Args:
        subdir: Directory to list. Can be absolute or relative to server start directory.
    """
    try:
        root = _safe(subdir)
    except ValueError as e:
        return err(str(e))

    if not os.path.isdir(root):
        return err(f"'{subdir}' is not a directory.", hint="Use list_files('.') for the root.")

    entries = []
    for name in sorted(os.listdir(root)):
        fp = os.path.join(root, name)
        entries.append({
            "name": name,
            "type": "dir" if os.path.isdir(fp) else "file",
            "size_kb": round(os.path.getsize(fp) / 1024, 2) if os.path.isfile(fp) else None,
        })

    return ok({
        "entries": entries,
        "count": len(entries),
        "root_dir": ROOT_DIR_ABS,
        "listed_dir": root,
    })


@mcp.tool(**tool_caps(
    caps=[CONTENT_WRITE, OVERWRITE, EDIT],
    fallbacks=["replace_in_file", "read_file_lines"],
    arg_roles={"edit_sig": ["content"]},
    risk_note="creates or overwrites files in the workspace",
    preview={"kind": "content", "args": ["content"]},
    label="Writing file: {path}",
))
def write_file(path: str, content: str, overwrite: bool = False) -> dict:
    """Create a file with the given content, or overwrite when overwrite=true.

    For existing files, prefer replace_in_file or replace_lines for surgical edits.
    Only use overwrite=true when you have read the file and intend to rewrite it entirely.

    Args:
        path:      ABSOLUTE path to the file (required; a relative path is rejected).
        content:   Text content to write.
        overwrite: Set to true to allow overwriting an existing file.
    """
    try:
        abs_err = _require_abs(path)
        if abs_err is not None:
            return abs_err
        fp = _safe(path)

        if os.path.exists(fp) and not os.path.isfile(fp):
            return err(f"'{path}' exists but is not a file.")

        is_existing = os.path.isfile(fp)
        if is_existing and not overwrite:
            return err(
                f"'{path}' already exists. Pass overwrite=true or use replace_in_file / replace_lines for surgical edits.",
                hint="Set overwrite=true if you intend to rewrite the entire file.",
            )

        syntax_error = _validate_python_content(path, content)
        if syntax_error:
            return err(
                syntax_error,
                hint=(
                    "Fix the syntax error before calling write_file again. "
                    "Common causes: unterminated string/docstring, missing colon after def/class, incorrect indentation."
                ),
            )

        _atomic_write_text(fp, content)
        operation = "updated" if is_existing else "created"

        return ok({
            "operation": operation,
            "path": path,
            "absolute_path": fp,
            "lines": _count_lines(content),
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(
    caps=[CONTENT_WRITE, EDIT],
    arg_roles={"edit_sig": ["content"]},
    risk_note="modifies files in the sandbox",
    preview={"kind": "append", "args": ["content"]},
    label="Appending to: {path}",
))
def append_file(path: str, content: str) -> dict:
    """Append text to a file (creates the file if it does not exist).

    For Python files, the final resulting content is syntax-checked before write.
    """
    try:
        abs_err = _require_abs(path)
        if abs_err is not None:
            return abs_err
        fp = _safe(path)

        if os.path.exists(fp) and not os.path.isfile(fp):
            return err(f"'{path}' exists but is not a file.")

        existing = _read_text(fp) if os.path.isfile(fp) else ""
        updated = existing + content

        syntax_error = _validate_python_content(path, updated)
        if syntax_error:
            return err(
                syntax_error,
                hint="The append would make the Python file invalid. Inspect the appended text and try again.",
            )

        _atomic_write_text(fp, updated)
        size_kb = round(os.path.getsize(fp) / 1024, 2)

        return ok({
            "operation": "appended",
            "path": path,
            "absolute_path": fp,
            "size_kb": size_kb,
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(
    caps=[REMOVE],
    reversibility=RECOVERABLE,
    fallbacks=["read_file_lines"],
    risk_note="deletes files in the sandbox",
    preview={"kind": "delete"},
    label="Deleting file: {path}",
))
def delete_file(path: str, confirm: bool = False) -> dict:
    """Delete a file immediately when confirm=True.

    Args:
        path: ABSOLUTE path to the file (required; a relative path is rejected).
        confirm: Must be True to perform the deletion.
    """
    try:
        abs_err = _require_abs(path)
        if abs_err is not None:
            return abs_err
        fp = _safe(path)
        if not os.path.isfile(fp):
            return err(f"'{path}' does not exist or is not a file.")
        if not confirm:
            return err(
                f"Deletion not performed for '{path}'.",
                hint="Call delete_file(path, confirm=True) to confirm deletion.",
            )

        os.remove(fp)
        return ok({
            "operation": "deleted",
            "path": path,
            "absolute_path": fp,
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(
    caps=[EDIT, REPLACEMENT_TRACK],
    arg_roles={"edit_sig": ["old_text", "new_text"]},
    risk_note="replaces content inside an existing file",
    preview={"kind": "replace", "args": ["old_text", "new_text"]},
    label="Editing file: {path}",
))
def replace_in_file(path: str, old_text: str, new_text: str) -> dict:
    """Replace exactly one occurrence of old_text with new_text in a file.

    CRITICAL — avoid duplicated code:
    - old_text must be the exact text to replace, unique in the file.
    - new_text must contain ONLY the replacement — never include lines that are
      already present in the file immediately before or after old_text.
      Including surrounding context lines in new_text is the #1 cause of
      duplicated code. Use more context in old_text instead to be unique.

    Example (correct):
      old_text = "    y = compute()"   # exactly the target line
      new_text = "    y = compute_v2()" # only what changes

    Example (WRONG — causes duplication):
      old_text = "    y = compute()"
      new_text = "    x = 1\n    y = compute_v2()\n    z = 3"  # includes context!

    The result includes a unified diff so you can verify the change is correct.
    If it looks wrong, issue another replace_in_file to repair it.

    Args:
        path:     ABSOLUTE path to the file (required; a relative path is rejected).
        old_text: Exact text to replace (must appear exactly once).
        new_text: Replacement text (must not include lines adjacent to old_text).
    """
    try:
        abs_err = _require_abs(path)
        if abs_err is not None:
            return abs_err
        fp = _safe(path)
        if not os.path.isfile(fp):
            return err(f"'{path}' does not exist or is not a file.")

        content = _read_text(fp)
        updated, error_msg, normalized = _apply_replace_in_text(content, old_text, new_text)
        if error_msg:
            # A benign re-issue of an already-applied edit reads as a no-op, not a
            # "Target text was not found" failure (see _edit_already_applied).
            if _edit_already_applied(content, old_text, new_text):
                return ok(_already_applied_result(path, fp))
            # The excerpt rides as its own field: err() collapses whitespace in the
            # message, and collapsed whitespace is unusable as an anchor.
            excerpt = _nearest_context(content, old_text)
            return err(error_msg, hint=_anchor_retry_hint(excerpt),
                       **({"anchor_excerpt": excerpt} if excerpt else {}))

        assert updated is not None  # for typing
        syntax_error = _validate_python_content(path, updated)
        if syntax_error:
            return err(syntax_error)

        old_lines = content.splitlines(True)
        new_lines = updated.splitlines(True)
        diff = _unified_diff(old_lines, new_lines, path)
        _atomic_write_text(fp, updated)
        result = {
            "operation": "replaced",
            "path": path,
            "absolute_path": fp,
            "replacements": 1,
            "diff": diff,
            **_edit_orientation(content, updated),
        }
        if normalized:
            result["normalized_match"] = True
            result["hint"] = (
                "Anchor matched after trailing-whitespace normalization. "
                "Update old_text to match the exact file content to avoid relying on fuzzy matching."
            )
        return ok(result)
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(
    caps=[EDIT],
    arg_roles={"edit_sig": ["start_line", "end_line", "new_content"]},
    preview={"kind": "line_splice", "args": ["start_line", "end_line", "new_content"]},
    label="Editing lines {start_line}–{end_line}: {path}",
))
def replace_lines(path: str, start_line: int, end_line: int, new_content: str) -> dict:
    """Replace a range of lines in a file with new content (1-based, inclusive).

    This is the primary way to make surgical edits to code — specify exactly
    which lines to replace. The line numbers must describe the file as it stands
    now: a previous edit's result reports the new numbering it produced, and a
    search reports the current one. Read the file when you have neither.

    Args:
        path:        ABSOLUTE path to the file (required; a relative path is rejected).
        start_line:  First line to replace (1-based).
        end_line:    Last line to replace (inclusive).
        new_content: Replacement text (may be fewer or more lines than the range).
                     Write the lines themselves, without the ``N: `` prefix a read adds.
    """
    try:
        abs_err = _require_abs(path)
        if abs_err is not None:
            return abs_err
        fp = _safe(path)
        if not os.path.isfile(fp):
            return err(f"'{path}' does not exist or is not a file.")

        content = _read_text(fp)
        old_lines = content.splitlines(True)
        # Replacement text pasted back from a numbered read, still numbered. Writing it
        # as given would put the line numbers into the file.
        new_content = _strip_line_number_prefixes(new_content, start_line) or new_content

        updated, error_msg = _apply_replace_lines_to_text(content, start_line, end_line, new_content)
        if error_msg:
            return err(error_msg, hint="Use read_file_lines to inspect the file and verify the exact line numbers.")

        assert updated is not None
        if updated == content:
            return ok({
                "operation": "noop",
                "path": path,
                "absolute_path": fp,
                "reason": "Lines already match the replacement content.",
            })

        syntax_error = _validate_python_content(path, updated)
        if syntax_error:
            # Give the agent enough context to understand what went wrong:
            # show the lines that were targeted and a repair hint.
            old_section = "".join(old_lines[start_line - 1 : end_line])
            hint = (
                f"The replacement would produce a syntax error — the file was NOT modified.\n"
                f"Lines {start_line}-{end_line} that you targeted contained:\n"
                f"'''\n{old_section}'''\n"
                f"Your new_content was:\n"
                f"'''\n{new_content}\n'''\n"
                f"Common causes: the range cuts through a docstring or multi-line "
                f"string, or the new_content has wrong indentation.\n"
                f"Fix: use replace_in_file with the exact text to match (it is "
                f"immune to line-number drift and handles multi-line strings safely)."
            )
            return err(syntax_error, hint=hint)

        new_lines = updated.splitlines(True)
        diff = _unified_diff(old_lines, new_lines, path)
        _atomic_write_text(fp, updated)

        return ok({
            "operation": "replace_lines",
            "path": path,
            "absolute_path": fp,
            "start_line": start_line,
            "end_line": end_line,
            "old_line_count": len(old_lines),
            "new_line_count": len(new_lines),
            "diff": diff,
            **_edit_orientation(content, updated),
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(
    caps=[EDIT, REPLACEMENT_TRACK],
    arg_roles={"confirm_gate": ["confirm"]},
    risk_note="replaces all occurrences of a pattern in a file",
    preview={"kind": "replace_all", "args": ["old_text", "new_text"]},
    label="Replacing in file: {path}",
))
def replace_all_in_file(
    path: str,
    old_text: str,
    new_text: str,
    whole_word: bool = True,
    confirm: bool = False,
) -> dict:
    """Replace ALL occurrences of old_text in a file.

    Safety features:
    - old_text must be at least 4 characters long.
    - By default whole_word=True:
      only matches at identifier boundaries are replaced.
      If old_text starts or ends with punctuation, only the relevant side is bounded.
      Example: "math." matches in "math.sin(...)".
    - By default confirm=False (dry-run): returns a preview and diff but does NOT write.
      Set confirm=True to apply after reviewing the preview.

    Typical two-step workflow:
      1. replace_all_in_file(path, "math.", "np.")                 -> preview
      2. replace_all_in_file(path, "math.", "np.", confirm=True)   -> apply
    """
    try:
        abs_err = _require_abs(path)
        if abs_err is not None:
            return abs_err
        fp = _safe(path)
        if not os.path.isfile(fp):
            return err(f"'{path}' does not exist or is not a file.")

        if len(old_text) < _MIN_REPLACE_ALL_PATTERN_LEN:
            return err(
                f"old_text is too short ({len(old_text)} chars). "
                f"Minimum length is {_MIN_REPLACE_ALL_PATTERN_LEN} to avoid accidental broad replacements.",
                hint="Use a longer, more specific pattern.",
            )

        content = _read_text(fp)
        occurrences = _find_all_occurrences(content, old_text, whole_word=whole_word)
        if not occurrences:
            # Re-issue of an already-applied replace-all: anchor gone, result present.
            if confirm and _edit_already_applied(content, old_text, new_text):
                return ok(_already_applied_result(path, fp))
            return err(
                "No occurrences found.",
                hint=(
                    "Read the file first to verify the exact text. "
                    "If using whole_word=True, check whether your pattern is inside a larger identifier."
                ),
            )

        updated, actual_count = _replace_all_in_text(content, old_text, new_text, whole_word=whole_word)
        old_lines = content.splitlines(True)
        new_lines = updated.splitlines(True)
        diff = _unified_diff(old_lines, new_lines, path)

        preview_matches = []
        for occ in occurrences[:30]:
            preview_matches.append({
                "line": occ["line"],
                "column": occ["column"],
                "context": occ["context"][:500],
            })

        if not confirm:
            return ok({
                "operation": "preview_replace_all",
                "path": path,
                "old_text": old_text,
                "new_text": new_text,
                "whole_word": whole_word,
                "match_count": actual_count,
                "matches": preview_matches,
                "diff": diff,
                "hint": (
                    f"Found {actual_count} occurrence(s). Review the diff and matches above, "
                    f"then call replace_all_in_file with confirm=True to apply."
                ),
            })

        syntax_error = _validate_python_content(path, updated)
        if syntax_error:
            return err(syntax_error)

        _atomic_write_text(fp, updated)
        return ok({
            "operation": "replace_all",
            "path": path,
            "absolute_path": fp,
            "old_text": old_text,
            "new_text": new_text,
            "whole_word": whole_word,
            "replacements": actual_count,
            "diff": diff,
            **_edit_orientation(content, updated),
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.resource(
    "files://list",
    name="files",
    description="Names of the files in the workspace root (attach with @files).",
)
def _list_resource() -> str:
    res = list_files()
    if not isinstance(res, dict) or res.get("status") != "ok":
        return ""

    entries = res.get("entries")
    if entries is None and isinstance(res.get("data"), dict):
        entries = res["data"].get("entries", [])

    if not isinstance(entries, list):
        return ""

    return "\n".join(
        e.get("name", "")
        for e in entries
        if isinstance(e, dict) and e.get("type") == "file"
    )


if __name__ == "__main__":
    mcp.run()
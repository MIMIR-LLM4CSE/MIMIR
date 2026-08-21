import ast
import bisect
import difflib
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok
from root_paths import resolve_path_in_root
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
    """None when *path* is absolute; an error payload naming the likely target otherwise.

    File tools take absolute paths only. A relative path has to be resolved against a
    root the model cannot see, and getting that wrong is silent: asked to create a file
    *outside* a directory, a model writes a bare relative name, the server resolves it
    against that very directory, and the run reports the constraint satisfied. No amount
    of prompt text made that inference reliable — removing the inference does.

    The suggestion is the point. Naming the path the relative form *would* have produced
    makes the rejection a one-step correction rather than an obstacle, and it states the
    workspace root at the moment placement is actually being decided, which no static
    prompt section can do.

    Callers must run this at the tool boundary, not inside :func:`_safe` — internal
    helpers (``list_files``) legitimately pass relative paths.
    """
    raw = "" if path is None else str(path).strip()
    if not raw:
        return err(f"Missing '{arg}'. File tools require an absolute path.")
    if os.path.isabs(os.path.expanduser(raw)):
        return None
    candidate = os.path.abspath(os.path.join(ROOT_DIR_ABS, os.path.normpath(raw)))
    return err(
        f"Relative {arg} '{raw}' — file tools require an absolute path.",
        hint=(f"Inside the workspace that is {candidate}. If you meant somewhere else, "
              f"give that absolute path instead. The workspace root is {ROOT_DIR_ABS}."),
    )


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
    """Return a short excerpt showing where *old_text* almost matches."""
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
    probe_lower = probe.lower()

    for idx, line in enumerate(lines):
        if probe_lower in line.lower():
            start = max(0, idx - 1)
            end = min(len(lines), idx + 2)
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
            msg += "\n" + ctx
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


def _apply_insert_in_text(content: str, after_text: str, new_text: str) -> tuple[str | None, str | None]:
    """Insert new_text after exactly one occurrence of after_text in content."""
    count = content.count(after_text)
    if count == 0:
        ctx = _nearest_context(content, after_text)
        msg = "Anchor text was not found."
        if ctx:
            msg += "\n" + ctx
        return None, msg
    if count > 1:
        return None, "Anchor text appears multiple times."
    match_pos = content.index(after_text)
    after_end = match_pos + len(after_text)
    # Check if new_text ends with text already immediately following after_text.
    after_context = content[after_end:]
    after_first = next(
        (ln.rstrip() for ln in after_context.splitlines() if ln.strip()),
        None,
    )
    new_lines_nonempty = [ln.rstrip() for ln in new_text.splitlines() if ln.strip()]
    if new_lines_nonempty and after_first and len(after_first.strip()) >= 8:
        if new_lines_nonempty[-1] == after_first:
            return None, (
                f"new_text ends with a line already present immediately after after_text: "
                f"{after_first.strip()!r}. "
                "new_text must contain ONLY the text to insert — "
                "do not include lines that are already in the file after the anchor. "
                "Remove the trailing context line(s) from new_text."
            )
    # Check if new_text starts with the after_text itself (common mistake: repeating the anchor).
    if new_text.lstrip("\n\r").startswith(after_text.lstrip("\n\r")):
        return None, (
            "new_text starts with the same content as after_text. "
            "after_text is the anchor already in the file — new_text is what comes AFTER it. "
            "Do not repeat the anchor text inside new_text."
        )
    return content.replace(after_text, after_text + new_text, 1), None


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
            return err(
                error_msg,
                hint="Use read_file_lines to inspect the current content and copy the exact whitespace and text for old_text.",
            )

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
    which lines to replace. Always read_file_lines first to confirm the exact
    line numbers.

    Args:
        path:        ABSOLUTE path to the file (required; a relative path is rejected).
        start_line:  First line to replace (1-based).
        end_line:    Last line to replace (inclusive).
        new_content: Replacement text (may be fewer or more lines than the range).
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
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))



@mcp.tool(**tool_caps(
    caps=[EDIT],
    arg_roles={"edit_batch": ["edits_json"], "confirm_gate": ["confirm"]},
    risk_note="applies a batch of edits to files in the workspace",
))
def apply_edits(edits_json: str, confirm: bool = False) -> dict:
    """Apply or preview a batch of file edits.

    When confirm=False (default), performs full validation and returns diffs
    WITHOUT modifying the filesystem.

    When confirm=True, applies the edits atomically per file.

    Notes:
    - Multiple edits targeting the same file are applied cumulatively in memory.
    - Writes are atomic per file (temp file + replace).
    - The batch is prevalidated, but not globally transactional across multiple files.

    Args:
        edits_json: A JSON array of edit objects. Each object must have:
            - "path": file path
            - "operation": one of replace_lines, replace_in_file, insert_in_file
            - operation-specific keys:
                * replace_lines: start_line, end_line, new_content
                * replace_in_file: old_text, new_text
                * insert_in_file: after_text, new_text
    """
    fingerprints: dict[str, tuple[float, int]] = {}
    
    try:
        edits = json.loads(edits_json)
    except (json.JSONDecodeError, TypeError) as e:
        return err(f"Invalid JSON: {e}", hint="Pass a JSON array of edit objects.")

    if not isinstance(edits, list) or not edits:
        return err("edits_json must be a non-empty JSON array.")

    originals: dict[str, str] = {}
    currents: dict[str, str] = {}
    path_map: dict[str, str] = {}
    per_edit_results: list[dict] = []
    
    # Track replace_lines ranges per file for overlap warnings
    line_ranges: dict[str, list[tuple[int, int]]] = {}
    warnings: list[str] = []

    # Pre-sort: for files with multiple replace_lines edits, reorder them so
    # the highest start_line is processed first (bottom-to-top application).
    # This ensures that earlier edits don't shift line numbers for later ones
    # when edits are applied cumulatively to currents[fp].
    _rl_positions: dict[str, list[int]] = {}
    for _i, _e in enumerate(edits):
        if isinstance(_e, dict) and str(_e.get("operation", "")) == "replace_lines":
            _rl_positions.setdefault(str(_e.get("path", "")).strip(), []).append(_i)
    for _positions in _rl_positions.values():
        if len(_positions) >= 2:
            _sorted_pos = sorted(
                _positions, key=lambda _j: -(int(edits[_j].get("start_line") or 0))
            )
            _saved = [edits[_k] for _k in _sorted_pos]
            for _slot, _orig in enumerate(_positions):
                edits[_orig] = _saved[_slot]

    # Phase 1: validate all edits in memory, cumulatively per file
    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return err(f"Edit #{idx}: not a JSON object.")

        path = str(edit.get("path", "")).strip()
        op = str(edit.get("operation", "")).strip()
        if not path:
            return err(f"Edit #{idx}: missing 'path'.")
        if not op:
            return err(f"Edit #{idx}: missing 'operation'.")

        # Same absolute-path rule as the single-file tools — a batch must not be a
        # way around it. Rejected in phase 1, so nothing is written.
        abs_err = _require_abs(path)
        if abs_err is not None:
            abs_err["error"] = f"Edit #{idx}: {abs_err.get('error', '')}"
            return abs_err

        try:
            fp = _safe(path)
        except ValueError as e:
            return err(f"Edit #{idx}: {e}")

        if not os.path.isfile(fp):
            return err(f"Edit #{idx}: '{path}' does not exist or is not a file.")

        if fp not in currents:
            # Capture original content
            content = _read_text(fp)
            originals[fp] = content
            currents[fp] = content
            path_map[fp] = path

            # Capture file fingerprint for TOCTOU protection
            try:
                stat = os.stat(fp)
                fingerprints[fp] = (stat.st_mtime, stat.st_size)
            except OSError as e:
                return err(
                    f"Edit #{idx} ({path}): cannot stat file for validation.",
                    hint=str(e),
                )

        current = currents[fp]

        if op == "replace_lines":
            try:
                start_line = int(edit.get("start_line"))
                end_line = int(edit.get("end_line"))
            except (TypeError, ValueError):
                return err(f"Edit #{idx} ({path}): start_line and end_line must be integers.")

            new_content = str(edit.get("new_content", ""))
            updated, error_msg = _apply_replace_lines_to_text(current, start_line, end_line, new_content)
            if error_msg:
                return err(f"Edit #{idx} ({path}): {error_msg}")
            
            # ── overlap tracking ──
            line_ranges.setdefault(fp, []).append((start_line, end_line))

        elif op == "replace_in_file":
            old_text = str(edit.get("old_text", ""))
            new_text = str(edit.get("new_text", ""))
            updated, error_msg, normalized = _apply_replace_in_text(current, old_text, new_text)
            if error_msg:
                # A sub-edit whose change is already present must not fail the whole
                # batch — record it as a no-op and move on.
                if _edit_already_applied(current, old_text, new_text):
                    per_edit_results.append({
                        "edit_index": idx, "path": path, "operation": op,
                        "status": "noop",
                        "reason": "already applied — original text absent, result present",
                    })
                    continue
                return err(f"Edit #{idx} ({path}): {error_msg}")

        elif op == "insert_in_file":
            after_text = str(edit.get("after_text", ""))
            new_text = str(edit.get("new_text", ""))
            updated, error_msg = _apply_insert_in_text(current, after_text, new_text)
            if error_msg:
                return err(f"Edit #{idx} ({path}): {error_msg}")

        else:
            return err(f"Edit #{idx}: unknown operation '{op}'.")

        assert updated is not None
        syntax_error = _validate_python_content(path, updated)
        if syntax_error:
            return err(f"Edit #{idx} ({path}): {syntax_error}")

        before_lines = current.splitlines(True)
        after_lines = updated.splitlines(True)
        edit_result: dict = {
            "edit_index": idx,
            "path": path,
            "operation": op,
            "status": "validated",
            "diff": _unified_diff(before_lines, after_lines, path),
        }
        if op == "replace_in_file" and normalized:
            edit_result["normalized_match"] = True
        per_edit_results.append(edit_result)

        currents[fp] = updated

    
    # Detect overlapping replace_lines edits (warnings only)
    for fp, ranges in line_ranges.items():
        if len(ranges) < 2:
            continue

        # Sort by start line
        sorted_ranges = sorted(ranges, key=lambda r: r[0])

        for (a_start, a_end), (b_start, b_end) in zip(
            sorted_ranges, sorted_ranges[1:]
        ):
            if b_start <= a_end:
                warnings.append(
                    f"Overlapping replace_lines edits in '{path_map[fp]}': "
                    f"lines {a_start}-{a_end} overlap with {b_start}-{b_end}."
                )

    file_previews: list[dict] = []
    
    # ── PREVIEW MODE ─────────────────────────────────────────────
    if not confirm:
        for fp, updated in currents.items():
            original = originals[fp]
            if updated == original:
                continue
            file_previews.append({
                "path": path_map[fp],
                "diff": _unified_diff(
                    original.splitlines(True),
                    updated.splitlines(True),
                    path_map[fp],
                ),
            })

        
        return ok({
            "operation": "preview_apply_edits",
            "edits_validated": len(per_edit_results),
            "files_touched": len(file_previews),
            "per_edit_results": per_edit_results,
            "file_previews": file_previews,
            "warnings": warnings,
            "hint": "Review the diffs and warnings above, then call apply_edits with confirm=true to apply.",
        })


    # Phase 2: write each changed file atomically
    file_results: list[dict] = []
    try:
        for fp, updated in currents.items():
            original = originals[fp]
            if updated == original:
                file_results.append({
                    "path": path_map[fp],
                    "status": "noop",
                    "diff": "",
                })
                continue
                        
            # TOCTOU check: ensure file has not changed since validation
            try:
                current_stat = os.stat(fp)
            except OSError as e:
                return err(
                    f"Failed to write '{path_map[fp]}': file no longer accessible.",
                    hint=str(e),
                )

            expected = fingerprints.get(fp)
            current = (current_stat.st_mtime, current_stat.st_size)

            if expected != current:
                return err(
                    f"File changed since validation: '{path_map[fp]}'.",
                    hint=(
                        "The file was modified externally between validation and write. "
                        "Re-run the operation to apply edits to the current version."
                    ),
                )

            _atomic_write_text(fp, updated)
            file_results.append({
                "path": path_map[fp],
                "status": "ok",
                "diff": _unified_diff(
                    original.splitlines(True),
                    updated.splitlines(True),
                    path_map[fp],
                ),
            })
    except Exception as write_err:
        already_written = [r["path"] for r in file_results if r["status"] == "ok"]
        return err(
            f"Batch write failed: {write_err}",
            hint="The batch was prevalidated, but a filesystem write failed. Already-written files are not rolled back automatically.",
            already_written=already_written,
        )

    
    return ok({
        "edits_validated": len(per_edit_results),
        "files_touched": len(file_results),
        "per_edit_results": per_edit_results,
        "file_results": file_results,
        "warnings": warnings,
    })



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
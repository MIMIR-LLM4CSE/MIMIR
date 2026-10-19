import os
from typing import Any


from mcp.server.fastmcp import FastMCP
from ...shared.atomic import write_text
from ...shared.responses import err, ok
from . import text_edits
from .root_paths import require_absolute, resolve_path_in_root
from ...shared.approved_roots import approved_roots
from ...shared.state_paths import standing_roots
from ...shared.capabilities import (
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe(path: str) -> str:
    """The absolute path of *path* inside the workspace, an approved root or the scratchpad."""
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


def _write_validated_text(fp: str, path: str, content: str) -> None:
    syntax_error = text_edits.validate_python_content(path, content)
    if syntax_error:
        raise ValueError(syntax_error)
    write_text(fp, content, newline="")



# ── tools ─────────────────────────────────────────────────────────────────────

# Not exposed as a tool — directory listing is owned by the search server's
# list_directory. Retained as a plain helper backing the files://list resource.
def list_files(subdir: str = ".") -> dict:
    """List the files and directories in *subdir*: ``{"entries": [{"name", "type", "size_kb"}], "count"}``."""
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


@mcp.tool(**tool_caps(kind="write",
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

        syntax_error = text_edits.validate_python_content(path, content)
        if syntax_error:
            return err(
                syntax_error,
                hint=(
                    "Fix the syntax error before calling write_file again. "
                    "Common causes: unterminated string/docstring, missing colon after def/class, incorrect indentation."
                ),
            )

        write_text(fp, content, newline="")
        operation = "updated" if is_existing else "created"

        return ok({
            "operation": operation,
            "path": path,
            "absolute_path": fp,
            "lines": text_edits.count_lines(content),
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(kind="write",
    caps=[CONTENT_WRITE, EDIT],
    arg_roles={"edit_sig": ["content"]},
    risk_note="modifies files in the sandbox",
    preview={"kind": "append", "args": ["content"]},
    label="Appending to: {path}",
))
def append_file(path: str, content: str) -> dict:
    """Append text to a file (creates the file if it does not exist).

    For Python files, the final resulting content is syntax-checked before write.

    Args:
        path:    File to append to, created if it does not exist.
        content: Text to add at the end. Appended verbatim — include a leading
                 newline if the file does not already end with one.
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

        syntax_error = text_edits.validate_python_content(path, updated)
        if syntax_error:
            return err(
                syntax_error,
                hint="The append would make the Python file invalid. Inspect the appended text and try again.",
            )

        write_text(fp, updated, newline="")
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


@mcp.tool(**tool_caps(kind="delete",
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


@mcp.tool(**tool_caps(kind="edit",
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
        updated, error_msg, normalized = text_edits.apply_replace_in_text(content, old_text, new_text)
        if error_msg:
            # A benign re-issue of an already-applied edit reads as a no-op, not a
            # "Target text was not found" failure (see edit_already_applied).
            if text_edits.edit_already_applied(content, old_text, new_text):
                return ok(text_edits.already_applied_result(path, fp))
            # The excerpt rides as its own field: err() collapses whitespace in the
            # message, and collapsed whitespace is unusable as an anchor.
            excerpt = text_edits.nearest_context(content, old_text)
            extra: dict[str, Any] = {"anchor_excerpt": excerpt} if excerpt else {}
            return err(error_msg, hint=text_edits.anchor_retry_hint(excerpt), **extra)

        assert updated is not None  # for typing
        syntax_error = text_edits.validate_python_content(path, updated)
        if syntax_error:
            return err(syntax_error)

        old_lines = content.splitlines(True)
        new_lines = updated.splitlines(True)
        diff = text_edits.unified_diff(old_lines, new_lines, path)
        write_text(fp, updated, newline="")
        result = {
            "operation": "replaced",
            "path": path,
            "absolute_path": fp,
            "replacements": 1,
            "diff": diff,
            **text_edits.edit_orientation(content, updated),
        }
        if normalized:
            result["normalized_match"] = True
            # note, not hint: `hint` is reserved and responses.ok() strips it from a
            # success payload, so this warning never reached the model.
            result["note"] = (
                "Anchor matched after trailing-whitespace normalization. "
                "Update old_text to match the exact file content to avoid relying on fuzzy matching."
            )
        return ok(result)
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(kind="edit",
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
        new_content = text_edits.strip_line_number_prefixes(new_content, start_line) or new_content

        updated, error_msg = text_edits.apply_replace_lines_to_text(content, start_line, end_line, new_content)
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

        syntax_error = text_edits.validate_python_content(path, updated)
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
        diff = text_edits.unified_diff(old_lines, new_lines, path)
        write_text(fp, updated, newline="")

        return ok({
            "operation": "replace_lines",
            "path": path,
            "absolute_path": fp,
            "start_line": start_line,
            "end_line": end_line,
            "old_line_count": len(old_lines),
            "new_line_count": len(new_lines),
            "diff": diff,
            **text_edits.edit_orientation(content, updated),
        })
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(kind="edit",
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

    Args:
        path:       File to rewrite.
        old_text:   Text to find. At least 4 characters.
        new_text:   Replacement text.
        whole_word: Match only at identifier boundaries (default True). Set False
                    to replace inside longer identifiers as well.
        confirm:    False (default) previews and writes nothing; True applies the
                    replacement. Preview first, then apply.
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
        occurrences = text_edits.find_all_occurrences(content, old_text, whole_word=whole_word)
        if not occurrences:
            # Re-issue of an already-applied replace-all: anchor gone, result present.
            if confirm and text_edits.edit_already_applied(content, old_text, new_text):
                return ok(text_edits.already_applied_result(path, fp))
            return err(
                "No occurrences found.",
                hint=(
                    "Read the file first to verify the exact text. "
                    "If using whole_word=True, check whether your pattern is inside a larger identifier."
                ),
            )

        updated, actual_count = text_edits.replace_all_in_text(content, old_text, new_text, whole_word=whole_word)
        old_lines = content.splitlines(True)
        new_lines = updated.splitlines(True)
        diff = text_edits.unified_diff(old_lines, new_lines, path)

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
                # note, not hint: `hint` is a reserved protocol key and
                # responses.ok() strips it from success payloads, so this line had
                # never once reached the model it tells what to do next.
                "note": (
                    f"Found {actual_count} occurrence(s). Review the diff and matches above, "
                    f"then call replace_all_in_file with confirm=True to apply."
                ),
            })

        syntax_error = text_edits.validate_python_content(path, updated)
        if syntax_error:
            return err(syntax_error)

        write_text(fp, updated, newline="")
        return ok({
            "operation": "replace_all",
            "path": path,
            "absolute_path": fp,
            "old_text": old_text,
            "new_text": new_text,
            "whole_word": whole_word,
            "replacements": actual_count,
            "diff": diff,
            **text_edits.edit_orientation(content, updated),
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
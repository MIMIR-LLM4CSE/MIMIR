from __future__ import annotations

import json
import os
import re
import shlex
from typing import Any, Awaitable, Callable

def is_python_test_filepath(path: str) -> bool:
    basename = os.path.basename(path)
    return path.endswith(".py") and (basename.startswith("test_") or basename.endswith("_test.py"))


def scratch_roots() -> list[str]:
    """The agent's scratchpad directories, as absolute real paths.

    Resolved from the client's own ``STATE_DIR``: ``MIMIR_STATE_DIR`` is placed
    only in the server subprocesses' environment, so calling the shared helper
    bare would resolve a different directory here than the sandbox uses.
    """
    from ...servers._shared.state_paths import standing_roots
    from ..config.constants import STATE_DIR
    return [os.path.realpath(os.path.abspath(r)) for r in standing_roots(STATE_DIR)]


def is_scratch_path(path: str) -> bool:
    """True if *path* lives in the scratchpad rather than the user's workspace.

    Scratch files are working material, not deliverables: they must not be
    recorded as produced work, demand validation, or appear in the change ledger.
    Without this the scratchpad would trade one problem (temporary files polluting
    the workspace) for another (temporary files polluting the completion state).
    """
    if not path:
        return False
    full = os.path.realpath(os.path.abspath(path))
    return any(
        full == root or full.startswith(root + os.sep) for root in scratch_roots()
    )


def absolute_workspace_path(path: str, *, cwd: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    root = cwd or os.getcwd()
    return os.path.abspath(os.path.join(root, path))


async def auto_validate_written_file(
    *,
    path: str,
    execution_context: dict[str, Any] | None,
    tool_owner: dict[str, str],
    run_tool: Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[str]],
    is_code_filepath: Callable[[str], bool],
    absolute_workspace_path_fn: Callable[[str], str],
) -> str:
    """Post-write completeness checks after a source-file write.

    The deterministic syntax→imports→lint→typecheck→tests validation *ladder* was
    removed: validating written code now happens through the bash server
    (``python -m py_compile`` / ``pytest`` / ``ruff`` / ``mypy``), steered by the
    validation guidance nudge rather than run automatically here. What remains are the
    two *completeness* checks that have no bash equivalent and never depended on a
    validator tool — a leftover ``old_text`` after a replace, and stale cross-file
    references after a workspace-wide rename. Best-effort and advisory only.
    """
    if execution_context is None or not path or not is_code_filepath(path):
        return ""

    absolute_path = absolute_workspace_path_fn(path)
    sections: list[str] = []

    # Post-edit completeness: grep for leftover old_text from the last replacement.
    leftover_warning = _check_replacement_completeness(absolute_path, execution_context)
    if leftover_warning:
        sections.append(f"COMPLETENESS_WARNING:\n{leftover_warning}")

    # Cross-file check: after replace_all_in_file, grep the workspace for the
    # old symbol in OTHER files that might still reference it.
    cross_file_warning = await _check_cross_file_references(
        edited_path=absolute_path,
        execution_context=execution_context,
        tool_owner=tool_owner,
        run_tool=run_tool,
    )
    if cross_file_warning:
        sections.append(f"CROSS_FILE_WARNING:\n{cross_file_warning}")

    return "\n\n".join(sections)


def _check_replacement_completeness(
    filepath: str,
    execution_context: dict[str, Any] | None,
) -> str:
    """After a replace edit, verify no leftover occurrences of old_text remain.

    Checks ``execution_context["last_replace_old_text"]`` (set by
    ``record_tool_observation`` for replace_in_file / replace_all_in_file).
    Returns a warning string if leftovers are found, or empty string.
    """
    if execution_context is None:
        return ""
    old_text = execution_context.get("last_replace_old_text")
    if not old_text or not os.path.isfile(filepath):
        return ""

    # Only check anchors that are specific enough to be structurally unique.
    # Short generic tokens (e.g. "path", "error", "value") appear throughout
    # source files for unrelated reasons and produce spurious warnings.
    # Require either multi-line content or a minimum visible length.
    _is_specific = "\n" in old_text or len(old_text.strip()) >= 20
    if not _is_specific:
        return ""

    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except OSError:
        return ""

    count = content.count(old_text)
    if count == 0:
        return ""

    # Skip warning when old_text is a substring of new_text: the replacement
    # itself re-introduces the pattern, so occurrences are expected and not
    # leftovers (e.g. replacing "compute" with "recompute").
    new_text = execution_context.get("last_replace_new_text") or ""
    if new_text and old_text in new_text:
        return ""

    # Find first few leftover locations for the warning
    lines = content.splitlines()
    locations: list[str] = []
    for line_no, line in enumerate(lines, 1):
        if old_text in line:
            locations.append(f"  line {line_no}: {line.strip()[:120]}")
        if len(locations) >= 5:
            break

    leftover_list = "\n".join(locations)
    return (
        f"WARNING: {count} leftover occurrence(s) of '{old_text}' still present in {filepath}.\n"
        f"The replacement may be incomplete. Remaining locations:\n{leftover_list}\n"
        f"Consider using replace_all_in_file to replace all occurrences, "
        f"or call replace_in_file again for each remaining match."
    )


# Directories excluded from the cross-file completeness grep — mirror of the
# search server's _SKIP_DIRS, so the bash grep does not scan .git/venv/build noise.
_CROSS_FILE_GREP_EXCLUDES = (
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
)
# ``old_text`` that is safe to hand to a read-only bash grep: identifier-ish text
# only. Anything with shell/regex-hostile characters is skipped — a non-read-only
# command would forfeit the approval exemption and prompt mid-validation, and the
# completeness check is a best-effort heuristic anyway.
_CROSS_FILE_SAFE_RE = re.compile(r"[\w.][\w.\- ]*")


async def _check_cross_file_references(
    *,
    edited_path: str,
    execution_context: dict[str, Any] | None,
    tool_owner: dict[str, str],
    run_tool: Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[str]],
) -> str:
    """After replace_all_in_file, grep the workspace for the old symbol in OTHER files.

    This catches cases where renaming a symbol in one file leaves callers in
    other files broken.  Only fires when ``cross_file_grep_old_text`` is set
    in the execution context (i.e. a confirmed replace_all_in_file just ran).

    Runs the search through the bash server's ``grep`` (there is no dedicated grep
    tool): a leading ``grep`` classifies read-only, so it is auto-exempt from
    approval and confined to the workspace. Gracefully no-ops if bash is
    unavailable or ``old_text`` is not a plain identifier.
    """
    if execution_context is None:
        return ""
    old_text = execution_context.pop("cross_file_grep_old_text", None)
    source_file = execution_context.pop("cross_file_grep_source", "")
    if not old_text or "bash_run" not in tool_owner:
        return ""
    if "\n" in old_text or not _CROSS_FILE_SAFE_RE.fullmatch(old_text):
        return ""

    excludes = " ".join(f"--exclude-dir={d}" for d in _CROSS_FILE_GREP_EXCLUDES)
    command = f"grep -rnI -F {excludes} -- {shlex.quote(old_text)} ."
    try:
        grep_result_str = await run_tool("bash_run", {"command": command}, execution_context)
    except Exception:
        return ""

    try:
        grep_payload = json.loads(grep_result_str) if isinstance(grep_result_str, str) else {}
    except (json.JSONDecodeError, TypeError):
        grep_payload = {}
    if not isinstance(grep_payload, dict) or grep_payload.get("status") != "ok":
        return ""
    stdout = grep_payload.get("stdout", "") or ""

    # Parse `path:line:text` lines, filter out the file we just edited.
    source_basename = os.path.basename(source_file) if source_file else ""
    other_file_matches: list[str] = []
    seen_files: set[str] = set()
    for row in stdout.splitlines():
        parts = row.split(":", 2)
        if len(parts) < 3:
            continue
        match_path, line, text = parts
        # `grep -r` prefixes hits with "./"; normalise for compare + display.
        if match_path.startswith("./"):
            match_path = match_path[2:]
        if not match_path:
            continue
        match_basename = os.path.basename(match_path)
        if source_file and (match_path == source_file or match_basename == source_basename):
            continue
        if match_path not in seen_files:
            seen_files.add(match_path)
            other_file_matches.append(f"  {match_path}:{line}: {text.strip()[:120]}")
        if len(other_file_matches) >= 10:
            break

    if not other_file_matches:
        return ""

    match_list = "\n".join(other_file_matches)
    return (
        f"WARNING: '{old_text}' still appears in {len(seen_files)} other file(s) after the rename.\n"
        f"These files may reference the old symbol and could break at runtime:\n{match_list}\n"
        f"Consider running replace_all_in_file on each affected file, or verify that these "
        f"occurrences are intentional (e.g. comments, strings, unrelated context)."
    )

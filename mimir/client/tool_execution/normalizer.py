from __future__ import annotations

from typing import Any

import os
from typing import Callable


TARGETED_LINE_READ_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".txt",
    ".rst",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sql",
    ".sh",
    ".bash",
    ".zsh",
    ".csv",
)

# The single ranged-read entrypoint, and the window its server applies when the caller
# names none. Mirrored here so a call can be made to say what it asks for before it
# leaves the client (see rewrite_tool_for_context).
RANGED_READ_TOOL = "read_file_lines"
DEFAULT_READ_WINDOW = 200

# Utility to recursively convert unhashable types (list → tuple, dict → frozenset) for cache keys
def _make_hashable(obj: Any) -> Any:
    """Recursively convert unhashable types (list → tuple, dict → frozenset) for cache keys."""
    if isinstance(obj, list):
        return tuple(_make_hashable(v) for v in obj)
    if isinstance(obj, dict):
        return frozenset((k, _make_hashable(v)) for k, v in obj.items())
    return obj


def normalize_workspace_path(path: str | None, *, cwd: str | None = None) -> str:
    if not path:
        return ""
    normalized = os.path.normpath(str(path).strip())
    if os.path.isabs(normalized):
        root = os.path.normpath(cwd or os.getcwd())
        if normalized == root or normalized.startswith(root + os.sep):
            return os.path.normpath(os.path.relpath(normalized, root))
    return normalized


def normalize_tool_path_argument(path: str | None, *, cwd: str | None = None) -> str:
    """Normalize path-like tool arguments to avoid duplicate-root mistakes."""
    raw = "" if path is None else str(path).strip()
    if not raw:
        return "."

    normalized = os.path.normpath(raw)
    root = os.path.normpath(cwd or os.getcwd())
    cwd_base = os.path.basename(root)

    # If the model passes the workspace basename, collapse to the root-relative dot path.
    if normalized == cwd_base:
        return "."
    prefix = cwd_base + os.sep
    if normalized.startswith(prefix):
        stripped = normalized[len(prefix):]
        return stripped or "."
    return normalized


def normalize_tool_arguments(
    tool_name: str,
    arguments: dict,
    *,
    path_args_by_tool: dict[str, tuple[str, ...]],
    normalize_tool_path_argument_fn: Callable[[str | None], str],
) -> dict:
    """Normalize known path-bearing arguments for discovery/read tools."""
    normalized = dict(arguments)
    for arg_name in path_args_by_tool.get(tool_name, ()):
        normalized[arg_name] = normalize_tool_path_argument_fn(normalized.get(arg_name))
    return normalized


def rewrite_tool_for_context(
    tool_name: str,
    arguments: dict,
    *,
    tool_owner: dict[str, str],
    is_code_filepath: Callable[[str], bool],
    normalize_workspace_path_fn: Callable[[str | None], str],
) -> tuple[str, dict]:
    """Rewrite broad tool calls into cheaper equivalents when safe."""
    if tool_name not in ("read_file", RANGED_READ_TOOL):
        return tool_name, arguments

    if RANGED_READ_TOOL not in tool_owner:
        return tool_name, arguments

    path = normalize_workspace_path_fn(arguments.get("path"))
    if not path:
        return tool_name, arguments

    # read_file is no longer a server-side tool — read_file_lines is the single
    # read entrypoint. For code/targeted-text we cap to a cheap leading window;
    # for everything else we read the whole file (end_line=0 ⇒ EOF).
    ext = os.path.splitext(path)[1].lower()
    is_targeted_text = is_code_filepath(path) or ext in TARGETED_LINE_READ_EXTENSIONS

    rewritten = dict(arguments)
    rewritten.setdefault("start_line", 1)
    if tool_name == RANGED_READ_TOOL:
        # A range the model left out is a range the *server* would pick. Filling it in
        # here is what makes the request say what it asks for: everything downstream
        # (coverage accounting, the repeat guards, the cache key) reads the arguments,
        # and a call that names no range is indistinguishable from one asking for the
        # whole file — which is how "read the file" kept meaning "read its header".
        rewritten.setdefault("end_line", DEFAULT_READ_WINDOW)
    elif is_targeted_text:
        rewritten.setdefault("end_line", 120 if is_code_filepath(path) else 160)
    else:
        rewritten.setdefault("end_line", 0)
    return RANGED_READ_TOOL, rewritten


def parent_path(path: str) -> str:
    normalized = os.path.normpath(path)
    parent = os.path.dirname(normalized)
    return parent if parent else "."

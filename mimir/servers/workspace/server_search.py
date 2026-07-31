"""
MCP Search Server
=================
Workspace reading and directory inspection, sandboxed to an allowed root
directory. The sandbox root defaults to the project codes directory but can be
overridden via the SEARCH_ROOT environment variable.

Tools: read_file_lines, read_files, list_directory, tree_summary. The File
server owns write and edit operations; this server owns reading and listing.
Text search is done with the bash server's `grep`/`rg` (classified read-only,
so it needs no approval and feeds the same discovery signals).
"""

import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, CACHEABLE, INSPECT_DIR, READ
from responses import err, ok
from root_paths import resolve_path_in_root
from approved_roots import approved_roots
from trusted_read_roots import trusted_read_roots

# Sandbox root — restrict all operations to this tree
SEARCH_ROOT = os.environ.get(
    "SEARCH_ROOT",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")  # codes/
    ),
)


def _extra_read_roots() -> list[str]:
    """Trusted out-of-workspace locations this read-only server may read from.

    Agent-produced run artefacts (proxy/HPC job logs, central session state) live
    outside the workspace; reading them is safe and was a recurring dead-end
    ("Path outside workspace"). This server is read/search-only, so widening its
    read roots never grants out-of-workspace writes (server_files stays strict).
    Extend at deploy time with MIMIR_EXTRA_FILE_ROOTS (os.pathsep-separated).
    """
    roots = trusted_read_roots()   # fixed caches + central state dir (shared source of truth)
    extra = os.environ.get("MIMIR_EXTRA_FILE_ROOTS", "")
    roots.extend(p for p in extra.split(os.pathsep) if p.strip())
    # Out-of-workspace paths the user approved this session (read/write/run).
    roots.extend(approved_roots())
    return roots

mcp = FastMCP(
    "SearchServer",
    debug=False,
    log_level="ERROR",
)

_MAX_FILE_SIZE = 2 * 1024 * 1024   # 2 MB — larger files are skipped
_TREE_CACHE: dict[str, dict] = {}
_TREE_CACHE_TTL = 60  # seconds
_SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".eggs", "node_modules",
    ".venv", "venv", "env", ".env",
    "dist", "build", "site-packages",
    ".mcp_backups", ".continue",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_root(path: str) -> str:
    """Resolve path inside SEARCH_ROOT or a trusted extra read root."""
    return resolve_path_in_root(path, SEARCH_ROOT, "search root",
                                extra_roots=_extra_read_roots())


def _tree_cache_key(path: str, max_depth: int, max_entries: int) -> str:
    # Normalize the path to avoid duplicate cache entries for equivalent paths
    # such as ".", "./", or an absolute path resolving to the same directory.
    try:
        normalized = _safe_root(path)
    except ValueError:
        normalized = path
    return f"{normalized}|{int(max_depth)}|{int(max_entries)}"


def _prune_tree_cache() -> None:
    """Best-effort removal of expired tree cache entries."""
    now = time.monotonic()
    expired = [
        key for key, value in _TREE_CACHE.items()
        if now - value.get("_mono", 0) >= _TREE_CACHE_TTL
    ]
    for key in expired:
        _TREE_CACHE.pop(key, None)
def _compute_tree_summary(root: str, max_depth: int, max_entries: int) -> tuple[str, int, bool]:
    lines = []
    truncated = False
    # Absolute, not a bare basename: rendered as a name, the root is
    # indistinguishable from a subdirectory, and a model asked to stay *out* of a
    # directory reads its own root as somewhere else. Same fix as
    # client/prompt/repo_baseline.py.
    root_name = os.path.abspath(root.rstrip(os.sep)) or "."
    for dirpath, dirnames, filenames in os.walk(root):
        rel_to_root = os.path.relpath(dirpath, root)
        depth = 0 if rel_to_root == "." else rel_to_root.count(os.sep) + 1
        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        indent = "  " * depth
        label = root_name if rel_to_root == "." else os.path.basename(dirpath)
        lines.append(f"{indent}{label}/ ({len(dirnames)} dirs, {len(filenames)} files)")
        if len(lines) >= max_entries:
            truncated = True
            break

        for fn in sorted(filenames):
            fp = os.path.join(dirpath, fn)
            size_kb = round(os.path.getsize(fp) / 1024, 1)
            lines.append(f"{indent}  {fn} [{size_kb} KB]")
            if len(lines) >= max_entries:
                truncated = True
                break
            if truncated:
                break

    return "\n".join(lines), len(lines), truncated


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[CACHEABLE, READ],
    path_args=['path'],
    arg_roles={"line_range": ["start_line", "end_line"]},
    label="Reading file: {path}",
))
def read_file_lines(path: str, start_line: int = 1, end_line: int = 200) -> dict:
    """Read a range of lines from a file inside the sandbox.

    Args:
        path:       File path relative to sandbox root.
        start_line: First line to return (1-based).
        end_line:   Last line to return (inclusive). Default 200. Pass end_line=0
                    (or negative) to read from start_line to the end of the file.
    """
    try:
        fpath = _safe_root(path)
    except ValueError as e:
        return err(str(e))
    if not os.path.isfile(fpath):
        return err(f"'{path}' is not a file.")
    if os.path.getsize(fpath) > _MAX_FILE_SIZE:
        return err("File is too large to read.", hint="Use grep() or narrow the requested file.")
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # SS-1: clamp start_line so 0/negative values don't index from the end.
        # end_line <= 0 is a sentinel meaning "read to EOF" (matches read_files).
        start_line = max(1, start_line)
        end_line = len(lines) if end_line <= 0 else max(start_line, end_line)
        selected = lines[start_line - 1: end_line]
        actual_end_line = start_line + len(selected) - 1 if selected else start_line - 1
        return ok({
            "path": fpath,
            "start_line": start_line,
            "end_line": actual_end_line,
            "content": "".join(selected),
            "lines_returned": len(selected),
        })
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(caps=[CACHEABLE, INSPECT_DIR], path_args=['path'],
                      label="Listing directory: {path}"))
def list_directory(path: str = ".") -> dict:
    """List the contents of a directory inside the sandbox.

    Args:
        path: Directory path relative to sandbox root.
    """
    try:
        full = _safe_root(path)
    except ValueError as e:
        return err(str(e))
    if not os.path.isdir(full):
        return err(f"'{path}' is not a directory.")
    entries = []
    for name in sorted(os.listdir(full)):
        fp = os.path.join(full, name)
        entries.append({
            "name": name,
            "path": fp,
            "type": "dir" if os.path.isdir(fp) else "file",
            "size_kb": round(os.path.getsize(fp) / 1024, 1) if os.path.isfile(fp) else None,
        })
    return ok({"entries": entries, "count": len(entries), "path": full})


@mcp.tool(**tool_caps(caps=[CACHEABLE, INSPECT_DIR], path_args=['path'],
                      label="Summarizing tree: {path}"))
def tree_summary(path: str = ".", max_depth: int = 2, max_entries: int = 120, use_cache: bool = True) -> dict:
    """Return a compact tree summary of a directory.

    Useful before creating a new file to understand the existing structure.

    Args:
        path: Directory path relative to sandbox root.
        max_depth: Maximum recursion depth from the requested path.
        max_entries: Maximum number of tree lines to return.
        use_cache: When true, return a cached summary if available; pass
            ``use_cache=False`` to force a fresh scan (e.g. after the structure changed).
    """
    try:
        root = _safe_root(path)
    except ValueError as e:
        return err(str(e))
    if not os.path.isdir(root):
        return err(f"'{path}' is not a directory.")

    _prune_tree_cache()
    key = _tree_cache_key(path, max_depth, max_entries)
    if use_cache and key in _TREE_CACHE:
        cached = _TREE_CACHE[key]
        if time.monotonic() - cached.get("_mono", 0) < _TREE_CACHE_TTL:
            return ok({
                "path": root,
                "tree": cached["tree"],
                "lines": cached["lines"],
                "truncated": cached["truncated"],
                "cached": True,
                "scanned_at": cached["scanned_at"],
            })
        else:
            del _TREE_CACHE[key]

    tree, lines_count, truncated = _compute_tree_summary(root, max_depth=max_depth, max_entries=max_entries)
    scanned_at = datetime.utcnow().isoformat() + "Z"
    _TREE_CACHE[key] = {
        "tree": tree,
        "lines": lines_count,
        "truncated": truncated,
        "scanned_at": scanned_at,
        "_mono": time.monotonic(),
    }

    return ok({
        "path": root,
        "tree": tree,
        "lines": lines_count,
        "truncated": truncated,
        "cached": False,
        "scanned_at": scanned_at,
    })


@mcp.tool(**tool_caps(caps=[CACHEABLE]))
def read_files(paths: list[str]) -> dict:
    """Read multiple files in a single call.

    Each file is read independently; an error in one file does not abort the
    others.  At most 10 paths are accepted per call.

    Args:
        paths:      List of file paths relative to sandbox root (max 10).
    """
    if not paths:
        return err("paths must be a non-empty list.")
    paths = paths[:10]
    start_line = 1
    end_line = 0

    files = []
    for path in paths:
        try:
            fpath = _safe_root(path)
        except ValueError as e:
            files.append({"path": path, "error": str(e)})
            continue
        if not os.path.isfile(fpath):
            files.append({"path": path, "error": "not a file"})
            continue
        if os.path.getsize(fpath) > _MAX_FILE_SIZE:
            files.append({"path": path, "error": "file too large"})
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            eff_end = len(lines) if end_line <= 0 else max(start_line, end_line)
            selected = lines[start_line - 1:eff_end]
            files.append({
                "path": fpath,
                "start_line": start_line,
                "end_line": start_line + len(selected) - 1,
                "total_lines": len(lines),
                "content": "".join(selected),
                "lines_returned": len(selected),
            })
        except OSError as e:
            files.append({"path": path, "error": str(e)})

    return ok({"files": files, "count": len(files)})


if __name__ == "__main__":
    mcp.run()

"""
MCP Search Server
=================
Workspace reading and directory inspection, sandboxed to an allowed root
directory. The sandbox root defaults to the project codes directory but can be
overridden via the SEARCH_ROOT environment variable.

Tools: read_file_lines, list_directory, tree_summary. The File
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
from root_paths import require_absolute, resolve_path_in_root
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
# Hard ceiling on one read, whatever range is asked for. Reading is targeted: a whole
# large file in the context is what stops the next read from fitting in the window.
_MAX_READ_LINES = 400
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
    # the client-side prompt builder.
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


def _number_lines(text: str, first_line: int) -> str:
    """*text* with every line prefixed by the number it has in the file.

    A read is the only thing that binds a line number to a line of text, and the edit
    tools are addressed by number — so a bare block leaves that binding to be counted by
    hand across the window. A model cannot do that reliably, and its way out is to keep
    re-reading narrower ranges until the requested range is one line and the number is
    no longer inferred but asserted. Same ``N: `` format as the search excerpts and the
    context window an edit hands back.
    """
    return "\n".join(
        f"{first_line + offset}: {line}"
        for offset, line in enumerate(text.splitlines())
    )


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[CACHEABLE, READ],
    path_args=['path'],
    label="Reading file: {path}",
))
def read_file_lines(path: str, start_line: int = 1, end_line: int = 200) -> dict:
    """Read a range of lines from a file inside the sandbox.

    At most _MAX_READ_LINES lines are returned per call, whatever the range asks for:
    reading is targeted here, and a whole large file in the context is what stops the
    next read from fitting. A window that stops short says so (`truncated`) and names
    where to resume (`next_start_line`).

    Every returned line carries its own number as a ``N: `` prefix, so the line to edit
    can be named from this reply alone. The prefix is not part of the file: strip it
    before copying text into an anchor.

    Args:
        path:       ABSOLUTE path to the file (required; a relative path is rejected).
        start_line: First line to return (1-based).
        end_line:   Last line to return (inclusive). Default 200. Pass end_line=0
                    (or negative) to read from start_line to the end of the file,
                    up to the per-call cap.
    """
    try:
        abs_err = require_absolute(path, SEARCH_ROOT)
        if abs_err is not None:
            return abs_err
        fpath = _safe_root(path)
    except ValueError as e:
        return err(str(e))
    if not os.path.isfile(fpath):
        return err(f"'{path}' is not a file.")
    if os.path.getsize(fpath) > _MAX_FILE_SIZE:
        return err("File is too large to read.",
                   hint="Search it with bash_run (grep -rn / rg) or read a narrower range.")
    try:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        # SS-1: clamp start_line so 0/negative values don't index from the end.
        # end_line <= 0 is a sentinel meaning "read to EOF".
        start_line = max(1, start_line)
        requested_end = len(lines) if end_line <= 0 else max(start_line, end_line)
        capped_end = min(requested_end, start_line + _MAX_READ_LINES - 1)
        selected = lines[start_line - 1: capped_end]
        actual_end_line = start_line + len(selected) - 1 if selected else start_line - 1
        payload = {
            "path": fpath,
            "start_line": start_line,
            "end_line": actual_end_line,
            "total_lines": len(lines),
            "content": _number_lines("".join(selected), start_line),
            "lines_returned": len(selected),
        }
        # Say so when the window stops short of the end. The default window makes
        # "read the file" mean "read its first 200 lines", and a caller that is not
        # told cannot tell the two apart — it just gets a header and reads again.
        if actual_end_line < len(lines):
            payload["truncated"] = True
            payload["next_start_line"] = actual_end_line + 1
        if capped_end < requested_end:
            payload["line_cap"] = _MAX_READ_LINES
        return ok(payload)
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

if __name__ == "__main__":
    mcp.run()

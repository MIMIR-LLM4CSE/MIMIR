"""
MCP Code-Intelligence Server
============================
Backend-abstracted code *navigation* for the agent: one tool per task, with the
server choosing the best available backend internally so the model never picks
between redundant tools.

Backend ladder (per call, best first):
    LSP language server  →  universal-ctags index  →  whole-word text scan

Every tier is optional. If neither a language server nor ``ctags`` is installed,
navigation tools return a structured error and the agent keeps using the search
server's grep/AST tools — no hard dependency (same graceful-degrade contract the
compile toolchain has when ``nvcc``/``gfortran`` is absent).

Tools (read-only):
    find_definition(name)        symbol  -> definition site(s)
    find_references(name)        symbol  -> use site(s)
    symbol_outline(path)         file    -> symbol tree
    hover(path, line, symbol)    LSP type/signature info (LSP-only)

The ``CODE_NAV`` capability exists so the client locates these navigation tools
by capability, never by literal name.
"""

import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import (
    tool_caps, CACHEABLE, CANDIDATE_SEARCH, CODE_NAV, READ,
    SEARCH, SEARCH_WITH_PATH,
)
from responses import err, ok
from root_paths import require_absolute, resolve_path_in_root
from approved_roots import approved_roots
from lsp_client import LSPPool, path_for

SEARCH_ROOT = os.environ.get(
    "SEARCH_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
)

mcp = FastMCP("CodeIntelServer", debug=False, log_level="ERROR")

_SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", ".eggs", "node_modules", ".venv", "venv", "env", ".env",
    "dist", "build", "site-packages", ".mcp_backups", ".continue",
}
_CODE_EXTS = (
    ".py", ".pyi", ".c", ".h", ".cpp", ".cc", ".cxx", ".hpp", ".hh",
    ".cu", ".cuh", ".f", ".f90", ".f95", ".f03", ".f08", ".for", ".fpp",
    ".js", ".ts", ".java", ".go", ".rs",
)
_MAX_TAGS = 60000
_MAX_RESULTS = 50
# Lines returned either side of a hit, and how many hits carry one. Enough to build an
# edit anchor without a follow-up read; capped so a 50-hit search stays a search.
_DEFAULT_CONTEXT_LINES = 3
_MAX_CONTEXT_LINES = 10
_CONTEXT_MAX_ENTRIES = 20
_INDEX_TTL = 60.0  # seconds; the ctags index auto-rebuilds once stale past this TTL

_lsp = LSPPool(SEARCH_ROOT)
# ctags index cache: {"defs": {name: [entry]}, "by_file": {rel: [entry]}, "_mono": t}
_index_cache: dict | None = None


# ── path helpers ──────────────────────────────────────────────────────────────

def _safe_root(path: str) -> str:
    return resolve_path_in_root(path, SEARCH_ROOT, "search root",
                                extra_roots=approved_roots())


def _out_path(path: str) -> str:
    """The form a navigation result is reported in: absolute.

    Every path the model reads should be one it can hand straight to a file tool,
    and those require absolute paths. Returning workspace-relative results would
    make the model join the root itself — reintroducing exactly the inference the
    absolute-path rule removed (see ``server_files._require_abs``), at the moment
    it is most likely to copy a path without thinking.

    The client normalizes these back to workspace-relative before storing them, so
    ``execution_context`` and every gate keep their existing form.
    """
    return os.path.abspath(path)


# ── ctags tier ────────────────────────────────────────────────────────────────

def _have_ctags() -> bool:
    from shutil import which
    return which("ctags") is not None


_CTAGS_EXCLUDES = [f"--exclude={d}" for d in _SKIP_DIRS]
# Field keys that carry a scope (parent symbol), used to indent the file outline.
_SCOPE_KEYS = frozenset({
    "class", "struct", "namespace", "union", "enum", "function", "module",
    "interface", "package", "method", "scope",
})


def _entry_from(name, path, line, kind, scope, signature, end=None) -> dict:
    abs_path = path if os.path.isabs(path) else os.path.join(SEARCH_ROOT, path)
    return {
        "name": name, "path": _out_path(abs_path),
        "line": int(line) if line else None,
        "end_line": int(end) if end else None,
        "kind": kind or "", "scope": scope or "", "signature": signature or "",
    }


def _parse_json_tags(stdout: str) -> dict | None:
    """Parse Universal-ctags ``--output-format=json`` output."""
    defs: dict[str, list] = {}
    by_file: dict[str, list] = {}
    n = 0
    for line in stdout.splitlines():
        if not line.startswith("{"):
            continue
        try:
            tag = json.loads(line)
        except ValueError:
            continue
        if tag.get("_type") != "tag" or not tag.get("name") or not tag.get("path"):
            continue
        e = _entry_from(tag["name"], tag["path"], tag.get("line"),
                        tag.get("kind", ""), tag.get("scope", ""), tag.get("signature", ""),
                        tag.get("end"))
        defs.setdefault(e["name"], []).append(e)
        by_file.setdefault(e["path"], []).append(e)
        n += 1
        if n >= _MAX_TAGS:
            break
    return {"defs": defs, "by_file": by_file} if defs else None


def _parse_traditional_tags(stdout: str) -> dict | None:
    """Parse Exuberant-ctags extended tags format (name<TAB>file<TAB>excmd;"<TAB>fields)."""
    defs: dict[str, list] = {}
    by_file: dict[str, list] = {}
    n = 0
    for line in stdout.splitlines():
        if not line or line.startswith("!_TAG_"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        name, path = parts[0], parts[1]
        kind = lineno = signature = scope = end = ""
        for field in parts[3:]:
            if ":" in field:
                key, val = field.split(":", 1)
                if key == "line":
                    lineno = val
                elif key == "end":
                    end = val
                elif key == "signature":
                    signature = val
                elif key in _SCOPE_KEYS:
                    scope = val
            elif field:
                kind = field  # bare extension field is the (full) kind name
        e = _entry_from(name, path, lineno, kind, scope, signature, end)
        defs.setdefault(name, []).append(e)
        by_file.setdefault(e["path"], []).append(e)
        n += 1
        if n >= _MAX_TAGS:
            break
    return {"defs": defs, "by_file": by_file} if defs else None


def _run_ctags(extra: list[str]) -> str:
    # +e carries each tag's end line: without it a symbol map says where every symbol
    # starts and never where one ends, which leaves "read the block around line N" to
    # be guessed a few lines at a time.
    cmd = ["ctags", "-R", "--fields=+nKsSe"] + _CTAGS_EXCLUDES + extra + [SEARCH_ROOT]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120, cwd=SEARCH_ROOT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout or ""


def _build_ctags_index() -> dict | None:
    """Build a symbol index from ctags, supporting Universal (JSON) and Exuberant.

    We don't trust the exit code (Exuberant prints "Unknown option" to stderr and
    may still exit 0); instead we try JSON first and fall back to the traditional
    extended format when JSON yields nothing. Either way returns a parsed index or
    None so the caller degrades to the text-scan tier.
    """
    if not _have_ctags():
        return None
    index = _parse_json_tags(_run_ctags(["--output-format=json", "-f", "-"]))
    if index is None:
        index = _parse_traditional_tags(_run_ctags(["-f", "-"]))
    if index is None:
        return None
    index["_mono"] = time.monotonic()
    return index


def _ctags_index(force: bool = False) -> dict | None:
    global _index_cache
    if (
        not force
        and _index_cache is not None
        and time.monotonic() - _index_cache.get("_mono", 0) < _INDEX_TTL
    ):
        return _index_cache
    built = _build_ctags_index()
    if built is not None:
        _index_cache = built
    elif force:
        _index_cache = None
    return _index_cache


# ── text-scan tier (always available; used for references fallback) ───────────

def _iter_code_files():
    for dirpath, dirnames, filenames in os.walk(SEARCH_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fn in filenames:
            if fn.lower().endswith(_CODE_EXTS):
                yield os.path.join(dirpath, fn)


def _whole_word_scan(name: str, *, max_results: int = _MAX_RESULTS) -> list[dict]:
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    hits: list[dict] = []
    for fpath in _iter_code_files():
        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if pattern.search(line):
                        hits.append({"path": _out_path(fpath), "line": i, "text": line.strip()[:200]})
                        if len(hits) >= max_results:
                            return hits
        except OSError:
            continue
    return hits


def _attach_context(entries: list[dict], context_lines: int) -> list[dict]:
    """Add a numbered excerpt around each hit's line, in place.

    A hit reported as ``{path, line}`` locates a symbol but cannot be acted on: an edit
    needs the surrounding text exactly as it stands, and a line number alone sends the
    caller back for a read whose answer this function already has open. Files are read
    once each and only for the entries that fit the budget.
    """
    if context_lines <= 0:
        return entries
    cache: dict[str, list[str]] = {}
    for entry in entries[:_CONTEXT_MAX_ENTRIES]:
        rel = entry.get("path")
        line = entry.get("line")
        if not rel or not isinstance(line, int) or line < 1:
            continue
        lines = cache.get(rel)
        if lines is None:
            try:
                with open(os.path.join(SEARCH_ROOT, rel), "r",
                          encoding="utf-8", errors="replace") as fh:
                    lines = fh.read().splitlines()
            except OSError:
                lines = []
            cache[rel] = lines
        if not lines or line > len(lines):
            continue
        lo = max(1, line - context_lines)
        hi = min(len(lines), line + context_lines)
        entry["context"] = {
            "start_line": lo,
            "end_line": hi,
            "text": "\n".join(f"{i}: {lines[i - 1]}" for i in range(lo, hi + 1)),
        }
    return entries


def _symbol_column(path_abs: str, line1: int, name: str) -> int:
    try:
        with open(path_abs, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if i == line1:
                    col = line.find(name)
                    return col if col >= 0 else 0
    except OSError:
        pass
    return 0


# ── LSP tier helpers ──────────────────────────────────────────────────────────

def _lsp_location_to_entry(loc: dict) -> dict | None:
    uri = loc.get("uri") or (loc.get("location") or {}).get("uri")
    rng = loc.get("range") or (loc.get("location") or {}).get("range")
    if not uri or not rng:
        return None
    start = rng.get("start", {})
    return {
        "path": _out_path(path_for(uri)),
        "line": (start.get("line", 0) or 0) + 1,
        "kind": _LSP_KIND.get(loc.get("kind"), ""),
        "name": loc.get("name", ""),
    }


_LSP_KIND = {
    5: "class", 6: "method", 9: "constructor", 12: "function",
    13: "variable", 14: "constant", 23: "struct", 11: "interface",
    10: "enum", 26: "type",
}


def _flatten_doc_symbols(symbols: list, out: list, depth: int = 0) -> None:
    for s in symbols or []:
        # Two shapes answer documentSymbol: the nested DocumentSymbol (its own
        # `range`) and the flat SymbolInformation (a `location.range`). Reading only
        # the first reported every symbol of a flat server at line 1.
        full = s.get("range") or (s.get("location") or {}).get("range") or {}
        rng = s.get("selectionRange") or full
        start = (rng.get("start") or {})
        # The end comes from the FULL range, never the selection range: the latter
        # covers the name alone, so its end would report a symbol one line long.
        end = ((full.get("end") or {}).get("line"))
        out.append({
            "name": s.get("name", ""),
            "kind": _LSP_KIND.get(s.get("kind"), ""),
            "line": (start.get("line", 0) or 0) + 1,
            "end_line": (end + 1) if isinstance(end, int) else 0,
            "depth": depth,
        })
        if s.get("children"):
            _flatten_doc_symbols(s["children"], out, depth + 1)


# ── tools ─────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[CODE_NAV, READ, CACHEABLE, SEARCH, SEARCH_WITH_PATH, CANDIDATE_SEARCH],
    label="Finding definition of {name}",
))
def find_definition(name: str, context_lines: int = _DEFAULT_CONTEXT_LINES) -> dict:
    """Locate where a symbol (function, class, type, variable) is DEFINED.

    Returns each definition site as {path, line, kind, signature, context}, where
    ``context`` is the surrounding source verbatim with line numbers — enough to build
    an edit anchor without a follow-up read. Prefers a language server's workspace
    symbol index when available, otherwise a universal-ctags index. Use this instead of
    grepping for a name when you want the authoritative definition.

    Args:
        name:          Symbol to locate.
        context_lines: Source lines to return either side of each hit (0 disables).
    """
    name = (name or "").strip()
    if not name:
        return err("A non-empty symbol name is required.")
    context_lines = max(0, min(int(context_lines or 0), _MAX_CONTEXT_LINES))

    # Tier 1: LSP workspace/symbol (any language server that is up).
    probe = os.path.join(SEARCH_ROOT, "__probe__.py")
    client = _lsp.client_for(probe)
    if client is not None:
        syms = client.workspace_symbol(name)
        if syms:
            entries = [
                e for e in (_lsp_location_to_entry(s) for s in syms)
                if e and e.get("name") == name
            ][:_MAX_RESULTS]
            if entries:
                return ok({"name": name, "backend": "lsp",
                           "definitions": _attach_context(entries, context_lines)})

    # Tier 2: ctags index.
    index = _ctags_index()
    if index is not None:
        entries = [
            {k: e[k] for k in ("path", "line", "kind", "signature")}
            for e in index["defs"].get(name, [])
        ][:_MAX_RESULTS]
        if entries:
            return ok({"name": name, "backend": "ctags",
                       "definitions": _attach_context(entries, context_lines)})
        return ok({"name": name, "definitions": [], "backend": "ctags",
                   "note": "No definition found in the symbol index."})

    return err(
        f"No code-intelligence backend available to locate '{name}'.",
        hint="Install universal-ctags or a language server, or use a text "
             "search (grep) to find the symbol.",
    )


@mcp.tool(**tool_caps(
    caps=[READ, CACHEABLE, SEARCH, SEARCH_WITH_PATH],
    label="Finding references to {name}",
))
def find_references(name: str, context_lines: int = 0) -> dict:
    """Find USE sites of a symbol across the workspace.

    Prefers a language server's reference resolution (anchored at the symbol's
    definition); falls back to a whole-word text scan. Returns {path, line, text} — the
    matching line itself, which locates a use site without quoting its surroundings.

    Args:
        name:          Symbol to locate.
        context_lines: Source lines to return either side of each hit. Off by default:
                       this answers "where is it used", and an excerpt per hit costs
                       more than the answer. Raise it when you must edit the sites.
    """
    name = (name or "").strip()
    if not name:
        return err("A non-empty symbol name is required.")
    context_lines = max(0, min(int(context_lines or 0), _MAX_CONTEXT_LINES))

    # Tier 1: LSP references anchored at a ctags/LSP-known definition position.
    index = _ctags_index()
    def_entry = None
    if index is not None and index["defs"].get(name):
        def_entry = index["defs"][name][0]
    if def_entry and def_entry.get("line"):
        abs_path = os.path.join(SEARCH_ROOT, def_entry["path"])
        client = _lsp.client_for(abs_path)
        if client is not None:
            col = _symbol_column(abs_path, int(def_entry["line"]), name)
            locs = client.references(abs_path, int(def_entry["line"]) - 1, col)
            if locs:
                refs = [e for e in (_lsp_location_to_entry(l) for l in locs) if e][:_MAX_RESULTS]
                if refs:
                    return ok({"name": name, "backend": "lsp",
                               "references": _attach_context(refs, context_lines)})

    # Tier 2: whole-word text scan (always available).
    hits = _whole_word_scan(name)
    return ok({"name": name, "references": _attach_context(hits, context_lines),
               "backend": "scan", "truncated": len(hits) >= _MAX_RESULTS})


def _file_line_count(abs_path: str) -> int:
    try:
        with open(abs_path, "rb") as fh:
            return sum(1 for _ in fh)
    except OSError:
        return 0


def _fill_end_lines(entries: list[dict], total_lines: int) -> list[dict]:
    """Give every symbol an ``end_line``, deriving the ones no backend reported.

    A symbol map that says only where each symbol *starts* leaves "read the block
    around line N" to be found by widening the window a few lines at a time. The end
    line is exact when the backend gives one (LSP ranges, universal-ctags ``+e``) and
    otherwise derived: a symbol runs until the next one at the same or shallower
    nesting, and the last one runs to the end of the file. That is an upper bound —
    trailing blank lines or module-level code between two symbols fall inside it — and
    an upper bound is the right error to make here: it costs a few extra lines in one
    read, where a short one costs another read.

    ``inferred_end`` marks which is which, since a derived end must not be taken for
    the exact boundary of the construct.
    """
    ordered = sorted(entries, key=lambda e: (int(e.get("line") or 0), int(e.get("depth") or 0)))
    for i, entry in enumerate(ordered):
        start = int(entry.get("line") or 0)
        if int(entry.get("end_line") or 0) >= start > 0:
            continue
        depth = int(entry.get("depth") or 0)
        end = total_lines
        for later in ordered[i + 1:]:
            if int(later.get("depth") or 0) <= depth:
                end = max(start, int(later.get("line") or 0) - 1)
                break
        entry["end_line"] = end or start
        entry["inferred_end"] = True
    return ordered


@mcp.tool(**tool_caps(
    caps=[CODE_NAV, READ, CACHEABLE],
    path_args=["path"],
    label="Outlining symbols in {path}",
))
def symbol_outline(path: str) -> dict:
    """List the symbols (classes, functions, types) defined in a single file.

    Returns an ordered outline [{name, kind, line, end_line, depth}] — read a symbol's
    whole span in one call rather than walking toward its end. ``end_line`` is exact
    when the backend reports one and otherwise derived from where the next symbol
    starts, in which case ``inferred_end`` is set and the span is an upper bound.
    Prefers a language server's document symbols; falls back to the ctags index.

    Args:
        path: ABSOLUTE path to the file (required; a relative path is rejected).
    """
    try:
        abs_err = require_absolute(path, SEARCH_ROOT)
        if abs_err is not None:
            return abs_err
        abs_path = _safe_root(path)
    except ValueError as e:
        return err(str(e))
    if not os.path.isfile(abs_path):
        return err(f"File not found: {path}", hint="Check the path and try again.")

    # Tier 1: LSP documentSymbol.
    client = _lsp.client_for(abs_path)
    if client is not None:
        syms = client.document_symbol(abs_path)
        if syms:
            out: list[dict] = []
            _flatten_doc_symbols(syms, out)
            if out:
                symbols = _fill_end_lines(out, _file_line_count(abs_path))
                return ok({"path": _out_path(abs_path),
                           "symbols": symbols,
                           "backend": "lsp"})

    # Tier 2: ctags index.
    index = _ctags_index()
    if index is not None:
        rel = _out_path(abs_path)
        entries = sorted(
            ({"name": e["name"], "kind": e["kind"], "line": e["line"] or 0,
              "end_line": e.get("end_line") or 0,
              "scope": e["scope"], "depth": 1 if e["scope"] else 0}
             for e in index["by_file"].get(rel, [])),
            key=lambda e: e["line"],
        )
        if entries:
            symbols = _fill_end_lines(entries, _file_line_count(abs_path))
            return ok({"path": rel, "symbols": symbols, "backend": "ctags"})
        return ok({"path": rel, "symbols": [], "backend": "ctags"})

    return err(
        f"No code-intelligence backend available to outline '{path}'.",
        hint="Install universal-ctags or a language server.",
    )


@mcp.tool(**tool_caps(caps=[READ, CACHEABLE], path_args=["path"],
                      label="Hovering {symbol} in {path}"))
def hover(path: str, line: int, symbol: str = "") -> dict:
    """Type / signature / doc info for a symbol at a position (LSP-only).

    ``path`` is an ABSOLUTE path (a relative one is rejected). ``line`` is 1-based.
    ``symbol`` (optional) is located on that line to compute the column. Returns the
    language server's hover text, or an error when no language server is available for
    the file's language.
    """
    try:
        abs_err = require_absolute(path, SEARCH_ROOT)
        if abs_err is not None:
            return abs_err
        abs_path = _safe_root(path)
    except ValueError as e:
        return err(str(e))
    if not os.path.isfile(abs_path):
        return err(f"File not found: {path}")
    client = _lsp.client_for(abs_path)
    if client is None:
        return err(
            f"No language server available for '{path}'.",
            hint="hover needs an LSP server (pyright/clangd/fortls). Use "
                 "find_definition or read_file_lines instead.",
        )
    col = _symbol_column(abs_path, int(line), symbol) if symbol else 0
    res = client.hover(abs_path, int(line) - 1, col)
    contents = (res or {}).get("contents") if isinstance(res, dict) else None
    text = ""
    if isinstance(contents, dict):
        text = contents.get("value", "")
    elif isinstance(contents, list):
        text = "\n".join(c.get("value", "") if isinstance(c, dict) else str(c) for c in contents)
    elif isinstance(contents, str):
        text = contents
    if not text:
        return ok({"path": _out_path(abs_path), "line": line, "hover": "", "backend": "lsp",
                   "note": "No hover information at this position."})
    return ok({"path": _out_path(abs_path), "line": line, "hover": text[:2000], "backend": "lsp"})


if __name__ == "__main__":
    mcp.run()

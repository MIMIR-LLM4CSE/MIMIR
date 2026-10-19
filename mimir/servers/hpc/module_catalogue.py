"""The site's module tree as a searchable catalogue, one per machine.

Hundreds to thousands of entries do not fit in a context window, so the tree is indexed
once and searched, never dumped: names synchronously (tier C), descriptions in the
background (Lmod's spider, then whatis), rebuilt when the tree's fingerprint changes.
Names rank first, then a lexical pool reranked semantically when embeddings exist.
"""

import hashlib
import json
import os
import re
import socket
import threading
import time
from functools import lru_cache
from typing import Any

from ...shared import embed as _embed
from ...shared.atomic import write_json
from ...shared.clock import utc_stamp
from ...shared.module_env import module_probe_script
from ...shared.state_paths import state_dir
from . import host_probe, vector_cache
from .platform_parsers import (
    parse_avail_terse,
    parse_spider_json,
    parse_whatis_bulk,
    parse_whatis_flat,
)

_CATALOGUE_SCHEMA = 1

# Named by hostname: STATE_DIR is per-workspace, but a module tree belongs to a
# machine. On a site whose state dir sits on a shared filesystem, a single file would
# be rebuilt on every hop between login nodes.
MODULES_DIR = os.environ.get(
    "MIMIR_MODULE_CATALOGUE_DIR", os.path.join(state_dir(), "platform", "modules")
)
CATALOGUE_FILE = os.path.join(MODULES_DIR, f"catalogue-{socket.gethostname()}.json")
EMBEDDINGS_FILE = os.path.join(MODULES_DIR, f"embeddings-{socket.gethostname()}.json")

# Lmod invocations for tier B cost ~30 ms each; past this many the catalogue keeps its
# name-only entries and says so rather than spending minutes on a 5000-module site.
_WHATIS_MAX = 800
# Bound on the stat walk behind the freshness signal.
_TREE_SCAN_MAX = 20000
# How many lexically-plausible modules get embedded for the semantic rerank. Memory
# embeds its whole corpus; a module catalogue must not, or the first search pays for
# thousands of texts.
SEMANTIC_POOL = 200
# The background enrichment's wall clock. Generous on purpose: nobody is waiting on it.
_ENRICH_BUDGET_SECS = int(os.environ.get("MIMIR_MODULE_INDEX_BUDGET", "600"))
# A manual refresh arriving inside this window is a no-op. Not an invalidation TTL —
# the signal alone decides freshness — just a debounce so a model that takes to
# passing refresh=True cannot stack enrichment threads.
_REFRESH_DEBOUNCE_SECS = 60

# Re-entrant so a test that runs the "background" work inline does not deadlock.
_BUILD_LOCK = threading.RLock()
_CATALOGUE_CACHE: dict | None = None
_ENRICHING = False
_LAST_BUILD_MONOTONIC: float | None = None


def schedule_background(fn) -> None:
    """Run *fn* off the request path. Replaced in tests to run it inline."""
    threading.Thread(target=fn, daemon=True).start()


@lru_cache(maxsize=1)
def module_flavour() -> str:
    """"lmod", "tmod", or "none" — no module system here, which is not an error."""
    script = module_probe_script(
        'if ! type module >/dev/null 2>&1; then echo none; '
        'elif [ -n "${LMOD_CMD:-}" ] || module --version 2>&1 | grep -qi lua; then echo lmod; '
        'else echo tmod; fi'
    )
    out = host_probe.run_shell(script, timeout=15)
    answer = (out["stdout"] or "").strip().splitlines()
    value = answer[-1].strip() if answer else ""
    return value if value in ("lmod", "tmod", "none") else "none"


@lru_cache(maxsize=1)
def effective_modulepath() -> str:
    """MODULEPATH as the site's init scripts leave it, not as it reaches this process.

    Memoized: this server never runs `module use` on itself, so the value cannot
    change while the process lives, and the check would otherwise spend a subprocess
    on every single search.
    """
    out = host_probe.run_shell(module_probe_script('echo "${MODULEPATH:-}"'), timeout=15)
    lines = [ln.strip() for ln in (out["stdout"] or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _lmod_cache_files() -> list[str]:
    """Lmod's own spider cache. Where it exists it is the most authoritative
    "the module tree changed" signal there is: the site regenerates it exactly then."""
    found = []
    roots = []
    mp_root = os.environ.get("MODULEPATH_ROOT", "")
    if mp_root:
        roots.append(os.path.join(os.path.dirname(mp_root.rstrip("/")), "cache"))
    roots.append(os.path.join(os.path.expanduser("~"), ".lmod.d", ".cache"))
    for root in roots:
        try:
            for name in sorted(os.listdir(root)):
                if name.startswith("spiderT"):
                    found.append(os.path.join(root, name))
        except OSError:
            continue
    return found


def _tree_fingerprint(modulepath: str) -> dict:
    """A deterministic hash of the module tree (directory mtimes two levels deep), never a timestamp."""
    parts: list[str] = []
    seen = 0
    truncated = False
    for root in [p for p in modulepath.split(":") if p]:
        try:
            st = os.stat(root)
        except OSError:
            continue
        parts.append(f"{root}|{st.st_mtime_ns}|{st.st_size}")
        try:
            families = sorted(os.scandir(root), key=lambda e: e.name)
        except OSError:
            continue
        for fam in families:
            if seen >= _TREE_SCAN_MAX:
                truncated = True
                break
            seen += 1
            try:
                parts.append(f"{fam.path}|{fam.stat().st_mtime_ns}")
                if not fam.is_dir():
                    continue
                for ver in sorted(os.scandir(fam.path), key=lambda e: e.name):
                    if seen >= _TREE_SCAN_MAX:
                        truncated = True
                        break
                    seen += 1
                    parts.append(f"{ver.path}|{ver.stat().st_mtime_ns}")
            except OSError:
                continue
        if truncated:
            break
    for cache_file in _lmod_cache_files():
        try:
            parts.append(f"{cache_file}|{os.stat(cache_file).st_mtime_ns}")
        except OSError:
            continue
    digest = hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()
    return {"tree": digest, "truncated": truncated}


def catalogue_signal(flavour: str) -> dict:
    modulepath = effective_modulepath()
    fingerprint = _tree_fingerprint(modulepath)
    return {
        "schema": _CATALOGUE_SCHEMA,
        "hostname": socket.gethostname(),
        "flavour": flavour,
        "modulepath": modulepath,
        **fingerprint,
    }


# ── tier collectors ───────────────────────────────────────────────────────────

def _collect_tier_c() -> list[dict]:
    """Names only. The floor: fast, and enough for a search by module name."""
    out = host_probe.run_shell(module_probe_script("module -t avail 2>&1"), timeout=30)
    return parse_avail_terse(out["stdout"] or "")


def _collect_tier_a(timeout: int) -> dict:
    """Lmod's spider. Never attempted on Tcl Environment Modules, which has no such
    command; the tier exists only where it exists."""
    modulepath = effective_modulepath()
    if not modulepath:
        return {}
    for fmt in ("jsonSoftwarePage", "spider-json"):
        script = module_probe_script(
            'SPIDER="${LMOD_DIR:-${LMOD_PKG:-}/libexec}/spider"; '
            f'[ -x "$SPIDER" ] || SPIDER=spider; "$SPIDER" -o {fmt} "$MODULEPATH" 2>/dev/null'
        )
        out = host_probe.run_shell(script, timeout=timeout)
        parsed = parse_spider_json(out["stdout"] or "")
        if parsed:
            return parsed
    return {}


def _collect_tier_b(flavour: str, names: list[str], timeout: int) -> dict:
    """Descriptions via whatis. One shell process either way, never one per module."""
    if not names:
        return {}
    if flavour == "tmod":
        out = host_probe.run_shell(module_probe_script("module whatis 2>&1"), timeout=timeout)
        return parse_whatis_flat(out["stdout"] or "", set(names))
    payload = "\n".join(names)
    script = module_probe_script(
        "while IFS= read -r m; do printf '@@%s\\n' \"$m\"; "
        "module whatis \"$m\" 2>&1; done <<'MIMIR_EOF'\n" + payload + "\nMIMIR_EOF"
    )
    out = host_probe.run_shell(script, timeout=timeout)
    return parse_whatis_bulk(out["stdout"] or "", names)



@lru_cache(maxsize=1)
def collect_modules() -> dict:
    """The modules loaded now and a pointer to the searchable catalogue (never builds it)."""
    flavour = module_flavour()
    if flavour == "none":
        return {"available": False, "module_system": "none",
                "note": "No Lmod or Tcl Environment Modules on this host."}
    loaded = host_probe.run_shell(module_probe_script("module -t list 2>&1"), timeout=15)
    catalogue = load_catalogue()
    summary = {"indexed": False}
    if catalogue:
        summary = {
            "indexed": True, "count": catalogue.get("count", 0),
            "tier": catalogue.get("tier", ""), "built_at": catalogue.get("built_at", ""),
            "partial": catalogue.get("partial", False),
        }
    return {
        "available": True,
        "module_system": flavour,
        "loaded": [e["load"] for e in parse_avail_terse(loaded["stdout"] or "")],
        "catalogue": summary,
        "note": "The module catalogue is not listed here — it is searchable with "
                "platform_search(query), by name ('cuda') or by capability "
                "('parallel hdf5'). Modules shown above are the ones loaded right now.",
    }


# ── catalogue persistence ─────────────────────────────────────────────────────

def load_catalogue() -> dict | None:
    """Read the catalogue from disk. Never builds — platform_probe depends on that."""
    try:
        with open(CATALOGUE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == _CATALOGUE_SCHEMA else None


def _save_catalogue(catalogue: dict) -> None:
    """A read-only state dir means rebuilding each session, not an error."""
    try:
        write_json(CATALOGUE_FILE, catalogue, ensure_ascii=False)
    except OSError:
        pass


def _empty_catalogue(flavour: str) -> dict:
    return {
        "schema": _CATALOGUE_SCHEMA, "built_at": utc_stamp(),
        "hostname": socket.gethostname(), "flavour": flavour,
        "tier": "none", "partial": False, "enrich_done": True,
        "signal": {}, "count": 0, "modules": [],
    }


def _enrich(base: dict) -> None:
    """Fill in descriptions behind the user's back, then swap the catalogue.

    Runs off the request path so nobody waits for spider. Its result is discarded if
    the tree moved while it worked — the next call will rebuild against what is
    actually there now.
    """
    global _ENRICHING, _CATALOGUE_CACHE, _LAST_BUILD_MONOTONIC
    try:
        started = time.monotonic()
        flavour = base.get("flavour", "")
        entries = [dict(e) for e in base.get("modules", [])]
        by_load = {e["load"]: e for e in entries}
        tier = "name"

        if flavour == "lmod":
            remaining = max(5, _ENRICH_BUDGET_SECS - int(time.monotonic() - started))
            for load, info in _collect_tier_a(min(120, remaining)).items():
                entry = by_load.get(load)
                if entry is None:
                    continue
                entry.update({
                    "description": info["description"],
                    "category": info["category"],
                    "keywords": info["keywords"],
                    "source": info["source"] or entry["source"],
                    "tier": "spider",
                })
                if info["default"]:
                    entry["default"] = True
                tier = "spider"

        pending = [e["load"] for e in entries if e["tier"] == "name"][:_WHATIS_MAX]
        if pending:
            remaining = max(5, _ENRICH_BUDGET_SECS - int(time.monotonic() - started))
            for load, description in _collect_tier_b(flavour, pending, remaining).items():
                entry = by_load.get(load)
                if entry is not None and description:
                    entry["description"] = description
                    entry["tier"] = "whatis"
                    if tier == "name":
                        tier = "whatis"

        if catalogue_signal(flavour) != base.get("signal"):
            return  # the tree moved under us; the next call rebuilds

        enriched = {
            **base,
            "built_at": utc_stamp(),
            "tier": tier,
            "partial": any(e["tier"] == "name" for e in entries),
            "enrich_done": True,
            "modules": entries,
            "count": len(entries),
        }
        with _BUILD_LOCK:
            _CATALOGUE_CACHE = enriched
            _LAST_BUILD_MONOTONIC = time.monotonic()
            _save_catalogue(enriched)
    except Exception:
        # A failed enrichment must never take the tier-C catalogue down with it.
        pass
    finally:
        _ENRICHING = False


def _build_tier_c(flavour: str, signal: dict) -> dict:
    entries = _collect_tier_c()
    return {
        "schema": _CATALOGUE_SCHEMA,
        "built_at": utc_stamp(),
        "hostname": socket.gethostname(),
        "flavour": flavour,
        "tier": "name",
        "partial": True,
        "enrich_done": False,
        "signal": signal,
        "count": len(entries),
        "modules": entries,
    }


def ensure_catalogue(refresh: bool = False) -> dict:
    """The catalogue for this host, building only what the caller must wait for.

    Synchronous work is tier C alone (under two seconds); descriptions arrive in the
    background. A hit on the in-process memo or on disk costs neither.
    """
    global _CATALOGUE_CACHE, _ENRICHING, _LAST_BUILD_MONOTONIC

    flavour = module_flavour()
    if flavour == "none":
        return _empty_catalogue(flavour)

    signal = catalogue_signal(flavour)
    with _BUILD_LOCK:
        debounced = (
            _LAST_BUILD_MONOTONIC is not None
            and (time.monotonic() - _LAST_BUILD_MONOTONIC) < _REFRESH_DEBOUNCE_SECS
        )
        forced = refresh and not debounced and not _ENRICHING

        catalogue = _CATALOGUE_CACHE
        if forced or catalogue is None or catalogue.get("signal") != signal:
            catalogue = None if forced else load_catalogue()
            if catalogue is not None and catalogue.get("signal") == signal:
                _CATALOGUE_CACHE = catalogue
            else:
                catalogue = _build_tier_c(flavour, signal)
                _CATALOGUE_CACHE = catalogue
                _LAST_BUILD_MONOTONIC = time.monotonic()
                _save_catalogue(catalogue)

        if not catalogue.get("enrich_done") and not _ENRICHING:
            _ENRICHING = True
            base = catalogue
            schedule_background(lambda: _enrich(base))

        return catalogue


# ── ranking ───────────────────────────────────────────────────────────────────

def _natural_version(version: str) -> tuple:
    """Sort 12.10 above 12.9, and never raise on a version that is not numbers."""
    chunks = []
    for part in re.findall(r"\d+|\D+", version or ""):
        if part.isdigit():
            chunks.append((0, int(part), ""))
        else:
            chunks.append((1, 0, part))
    return tuple(chunks)


def _search_text(entry: dict) -> str:
    return " ".join([
        entry.get("name", ""), entry.get("version", ""),
        entry.get("description", ""), entry.get("category", ""),
        " ".join(entry.get("keywords", [])),
    ]).strip()


def name_matches(modules: list[dict], query: str) -> list[dict]:
    """Exact, then prefix, then substring; defaults first, newest version first.

    This layer is what makes a description-free, embedding-free catalogue genuinely
    useful rather than merely non-broken, so it runs first and always.
    """
    q = query.strip().lower()
    if not q:
        return []
    ranked = []
    for entry in modules:
        name = entry.get("name", "").lower()
        load = entry.get("load", "").lower()
        if name == q or load == q:
            rank = 0
        elif name.startswith(q) or load.startswith(q):
            rank = 1
        elif q in name or q in load:
            rank = 2
        else:
            continue
        ranked.append((rank, entry))
    ranked.sort(key=lambda pair: _natural_version(pair[1].get("version", "")), reverse=True)
    ranked.sort(key=lambda pair: (pair[0], 0 if pair[1].get("default") else 1))
    return [entry for _rank, entry in ranked]


def rank_modules(modules: list[dict], query: str, limit: int) -> list[dict]:
    results: list[dict] = []
    seen: set[str] = set()

    for entry in name_matches(modules, query):
        if entry["key"] in seen:
            continue
        seen.add(entry["key"])
        results.append({**entry, "score": None, "match": "name"})
        if len(results) >= limit:
            return results

    described = [e for e in modules if e.get("description") or e.get("keywords")]
    if not described:
        return results

    texts = [_search_text(e) for e in described]
    pool = [described[idx] for idx, overlap in _embed.lexical_rank(query, texts)
            if overlap > 0][:SEMANTIC_POOL]
    if not pool:
        return results

    semantic = vector_cache.semantic_rank(
        query, pool,
        key_of=lambda e: e["key"],
        text_of=_search_text,
        path=EMBEDDINGS_FILE,
        limit=limit,
    )
    ordered: list[tuple[Any, float | None, str]]
    if semantic is not None:
        ordered = [(entry, score, "semantic") for entry, score in semantic]
    else:
        ordered = [(entry, None, "lexical") for entry in pool]

    for entry, score, kind in ordered:
        if entry["key"] in seen:
            continue
        seen.add(entry["key"])
        results.append({**entry, "score": score, "match": kind})
        if len(results) >= limit:
            break
    return results


def enriching() -> bool:
    """Whether descriptions are being filled in behind the current catalogue."""
    return _ENRICHING


def reset() -> None:
    """Forget everything this process holds about the catalogue (the file stays)."""
    global _CATALOGUE_CACHE, _ENRICHING, _LAST_BUILD_MONOTONIC
    module_flavour.cache_clear()
    effective_modulepath.cache_clear()
    collect_modules.cache_clear()
    with _BUILD_LOCK:
        _CATALOGUE_CACHE = None
        _ENRICHING = False
        _LAST_BUILD_MONOTONIC = None

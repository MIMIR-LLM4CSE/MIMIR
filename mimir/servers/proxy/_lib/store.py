"""The proxy server's persistence over the layout of ``shared/proxy_store``: the registry,
suite, reference and optimization-session stores, and their locks.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import threading

from ....shared.atomic import write_json, write_text
from ....shared.proxy_store import (
    active_session_file, cache_dir, opt_config_file, opt_session_runs_dir,
    read_json, registry_path, resolve_proxy_name,
)
from ....shared.responses import err

# ── storage root ──────────────────────────────────────────────────────────────

def workspace_root() -> str:
    """The live workspace root, realpath-resolved (``build`` keeps its spelling instead)."""
    return os.path.realpath(os.path.abspath(
        os.environ.get("MCP_FILES_ROOT") or os.getcwd()))


def registry_lock_path() -> str:
    return registry_path() + ".lock"


def refs_dir() -> str:
    return os.path.join(cache_dir(), "references")


def runs_dir() -> str:
    return os.path.join(cache_dir(), "runs")


def suites_dir() -> str:
    return os.path.join(cache_dir(), "suites")


def scaffolds_dir() -> str:
    return os.path.join(cache_dir(), "scaffolds")


# ── run directories ───────────────────────────────────────────────────────────

def run_dir_names(parent: str) -> list[str]:
    """Run-dir names under *parent*, newest first; the 'active' symlink excluded."""
    if not os.path.isdir(parent):
        return []
    return sorted(
        [d for d in os.listdir(parent)
         if d != "active" and os.path.isdir(os.path.join(parent, d))],
        reverse=True,
    )


# ── locking ───────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def file_lock(lock_path: str):
    """Exclusive flock on *lock_path*, creating its directory if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(lock_path)), exist_ok=True)
    with open(lock_path, "w") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)


# ── registry ──────────────────────────────────────────────────────────────────

_REGISTRY_LOCK_LOCAL = threading.local()


@contextlib.contextmanager
def registry_lock(create: bool = False):
    """A re-entrant (thread-local depth) flock around registry read-modify-write.

    Without a store on disk and without *create*, it does nothing: a read must not create
    ``proxy_bench/``. Only ``register`` passes ``create=True``.
    """
    depth = getattr(_REGISTRY_LOCK_LOCAL, "depth", 0)
    if depth > 0:
        _REGISTRY_LOCK_LOCAL.depth = depth + 1
        try:
            yield
        finally:
            _REGISTRY_LOCK_LOCAL.depth -= 1
        return
    if not create and not os.path.isdir(cache_dir()):
        yield
        return
    os.makedirs(cache_dir(), exist_ok=True)
    fd = open(registry_lock_path(), "w")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _REGISTRY_LOCK_LOCAL.depth = 1
        yield
    finally:
        _REGISTRY_LOCK_LOCAL.depth = 0
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()


def load_registry() -> dict:
    # Always acquire the registry lock for all registry reads (even read-only)
    with registry_lock():
        path = registry_path()
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    return json.load(fh)
            except json.JSONDecodeError:
                backup = path + ".corrupt"
                try:
                    shutil.copy2(path, backup)
                except OSError:
                    pass
                raise RuntimeError(
                    f"registry.json is corrupt (invalid JSON). "
                    f"A backup was saved to {backup}. "
                    "Delete or fix the backup, then re-register your proxies."
                )
            except OSError:
                pass
        return {}


def load_registry_or_err() -> tuple[dict, str | None]:
    """Return (registry, None) on success or ({}, error_message) on corruption."""
    try:
        return load_registry(), None
    except RuntimeError as exc:
        return {}, str(exc)


def save_registry(reg: dict) -> None:
    # Always acquire the registry lock for all registry writes. No ``create``: a save
    # is always reached either under an outer lock that already took it (register) or
    # on an entry that was found, which means the store is already there. The atomic
    # write creates the directory itself in any case.
    with registry_lock():
        write_json(registry_path(), reg, indent=2)


# ── run / reference layout ────────────────────────────────────────────────────

def proxy_runs_dir(proxy_name: str) -> str:
    return os.path.join(runs_dir(), proxy_name)


def active_link(proxy_name: str) -> str:
    return os.path.join(proxy_runs_dir(proxy_name), "active")


def reference_dir(reference_name: str) -> str:
    return os.path.join(refs_dir(), reference_name)


def load_ref_metrics(reference_name: str) -> dict | None:
    return read_json(os.path.join(reference_dir(reference_name), "metrics.json"))


def ref_output_path(reference_name: str) -> str | None:
    rd = reference_dir(reference_name)
    for ext in (".npz", ".dat"):
        p = os.path.join(rd, f"output{ext}")
        if os.path.isfile(p):
            return p
    return None


# ── suites ────────────────────────────────────────────────────────────────────

def suite_path(name: str) -> str:
    return os.path.join(suites_dir(), name, "suite.json")


def load_suite(name: str) -> dict | None:
    return read_json(suite_path(name))


def save_suite(name: str, suite: dict) -> None:
    write_json(suite_path(name), suite, indent=2)


def suite_results_dir(name: str) -> str:
    return os.path.join(suites_dir(), name, "results")


def latest_suite_results(name: str) -> str | None:
    rd = suite_results_dir(name)
    if not os.path.isdir(rd):
        return None
    entries = sorted(
        [d for d in os.listdir(rd) if os.path.isdir(os.path.join(rd, d))],
        reverse=True,
    )
    return os.path.join(rd, entries[0]) if entries else None


# ── optimization-session layout ───────────────────────────────────────────────


def opt_active_link(proxy_name: str) -> str:
    return os.path.join(opt_session_runs_dir(proxy_name), "active")


def opt_ledger_file(proxy_name: str) -> str:
    """Append-only ledger of every completed run: opt_runs/<proxy>/ledger.jsonl."""
    return os.path.join(opt_session_runs_dir(proxy_name), "ledger.jsonl")


def opt_best_file(proxy_name: str) -> str:
    """Best-so-far pointer: opt_runs/<proxy>/best.json."""
    return os.path.join(opt_session_runs_dir(proxy_name), "best.json")


def opt_git_dir() -> str:
    """Shadow repository for the tracked tree, inside the proxy store."""
    return os.path.join(cache_dir(), "opt.git")


def load_opt_config(proxy_name: str = "") -> dict:
    name = resolve_proxy_name(proxy_name)
    if not name:
        return {}
    return read_json(opt_config_file(name), {}) or {}


def save_opt_config(cfg: dict, proxy_name: str = "") -> None:
    name = proxy_name or cfg.get("proxy_name", "")
    if not name:
        return
    write_json(opt_config_file(name), cfg, indent=2)


def entry_or_err(proxy_name: str) -> tuple[dict, dict | None]:
    """A registered proxy's entry, or ``{}`` and the ready-to-return error saying why not.

    The same four lines stood in five ops with the same wording; a sixth caller
    would have copied them again.
    """
    reg, reg_err = load_registry_or_err()
    if reg_err:
        return {}, err(reg_err)
    if proxy_name not in reg:
        return {}, err(f"Proxy '{proxy_name}' not registered.",
                         hint="Call proxy_manage(op='register', ...) first.")
    return reg[proxy_name], None


def write_active_session(proxy_name: str, session_id: str | None = None) -> None:
    """Persist *proxy_name* as this MIMIR session's most-recently initialized proxy."""
    write_text(active_session_file(session_id), proxy_name)


def clear_active_session(session_id: str | None = None) -> None:
    """Drop the active-session pointer (best-effort). Leaves opt_config/runs intact
    for history; only the "current session" marker is removed, so nameless ops no
    longer resolve to it and the client's proxy-exec guard lifts."""
    try:
        os.remove(active_session_file(session_id))
    except FileNotFoundError:
        pass
    except OSError:
        pass

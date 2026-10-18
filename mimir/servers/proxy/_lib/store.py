"""Where proxy state lives (under ``<workspace>/proxy_bench/``) and its persistence.

Every path derives from ``_CACHE_DIR`` (``MIMIR_PROXY_BENCH_DIR``); also the atomic IO and
the registry, suite, reference and optimization-session stores.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import threading

from ..._shared.atomic import write_json, write_text
from ..._shared.responses import err

# ── storage root ──────────────────────────────────────────────────────────────

# Under the workspace, so the experiment goes with the project; not `.mimir/` (the
# extensions directory) nor the state dir (references are megabytes). Workspace from
# MCP_FILES_ROOT, else cwd; MIMIR_PROXY_BENCH_DIR wins (tests).
_WORKSPACE_ROOT = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())

_CACHE_DIR = (os.environ.get("MIMIR_PROXY_BENCH_DIR")
              or os.path.join(_WORKSPACE_ROOT, "proxy_bench"))


def cache_dir() -> str:
    return _CACHE_DIR


def workspace_root() -> str:
    """The live workspace root, realpath-resolved (``build`` keeps its spelling instead)."""
    return os.path.realpath(os.path.abspath(
        os.environ.get("MCP_FILES_ROOT") or os.getcwd()))


def registry_path() -> str:
    return os.path.join(_CACHE_DIR, "registry.json")


def registry_lock_path() -> str:
    return registry_path() + ".lock"


def refs_dir() -> str:
    return os.path.join(_CACHE_DIR, "references")


def runs_dir() -> str:
    return os.path.join(_CACHE_DIR, "runs")


def suites_dir() -> str:
    return os.path.join(_CACHE_DIR, "suites")


def scaffolds_dir() -> str:
    return os.path.join(_CACHE_DIR, "scaffolds")


def opt_runs_dir() -> str:
    return os.path.join(_CACHE_DIR, "opt_runs")


def active_session_file(session_id: str | None = None) -> str:
    """The pointer to the proxy a nameless ``proxy_eval`` op acts on: one per MIMIR session."""
    sid = session_id if session_id is not None else _mimir_session_id()
    name = f"active_session.{sid}" if sid else "active_session"
    return os.path.join(opt_runs_dir(), name)


def _mimir_session_id() -> str:
    """The MIMIR session this process is acting for, or "".

    Best-effort: a store reachable without the shared state helpers (a bare test
    fixture) must keep working, and "no session" is the pre-existing behaviour.
    """
    try:
        from ..._shared.state_paths import active_session_id
        return active_session_id()
    except Exception:
        return ""


# ── generic atomic IO ─────────────────────────────────────────────────────────

def read_json(path: str, default=None):
    """Best-effort JSON read; returns *default* on a missing or corrupt file."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


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
    if not create and not os.path.isdir(_CACHE_DIR):
        yield
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
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

def opt_session_runs_dir(proxy_name: str) -> str:
    """Directory that holds timestamped run dirs for one proxy's opt sessions."""
    return os.path.join(opt_runs_dir(), proxy_name)


def opt_config_file(proxy_name: str) -> str:
    """Per-proxy opt config path: opt_runs/<proxy_name>/opt_config.json."""
    return os.path.join(opt_session_runs_dir(proxy_name), "opt_config.json")


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


def resolve_proxy_name(arg: str, session_id: str | None = None) -> str | None:
    """Return *arg* if non-empty, else read this session's pointer, else None.

    *session_id* is for callers in the client process, where N sessions share one
    environment and the id can only come from the agent that is acting.
    """
    if arg:
        return arg
    path = active_session_file(session_id)
    if os.path.isfile(path):
        try:
            name = open(path).read().strip()
            if name:
                return name
        except OSError:
            pass
    return None


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

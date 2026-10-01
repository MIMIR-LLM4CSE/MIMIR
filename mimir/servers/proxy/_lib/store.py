"""Storage layout and persistence for the proxy server.

Single source of truth for where proxy state lives on disk.  Every path is
derived from the one module attribute ``_CACHE_DIR`` (env-overridable via
``MIMIR_PROXY_BENCH_DIR``), so tests repoint exactly one variable to get a
hermetic store.  It defaults under the *workspace*, so the state of an experiment
lives with the project it belongs to.  Also owns the generic atomic-IO helpers and the registry /
suite / reference / optimization-session persistence built on them.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import shutil
import threading

from responses import err

# ── storage root ──────────────────────────────────────────────────────────────

# Under the WORKSPACE, not ~/.cache. Everything else MIMIR persists is already scoped
# per workspace; this store was the exception, and the exception had a cost: deleting a
# project left the registry behind with a run_cmd_template pointing at a file that no
# longer existed, and an `opt_runs/active_session` still naming an optimisation as "in
# progress" — so a fresh start silently resumed the old one. Here it is visible, and it
# goes when the project goes.
#
# Not `<workspace>/.mimir/`: that name is the workspace's *extensions* directory. Not the
# per-workspace state dir either: sealed reference fields are npz of several megabytes
# each, and ~/.mimir has no business growing without bound.
#
# MCP_FILES_ROOT is how every server here learns the workspace root (see server_bash);
# cwd is its fallback. MIMIR_PROXY_BENCH_DIR still wins, which is what lets a test
# repoint the whole store by setting one variable.
_WORKSPACE_ROOT = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())

_CACHE_DIR = (os.environ.get("MIMIR_PROXY_BENCH_DIR")
              or os.path.join(_WORKSPACE_ROOT, "proxy_bench"))


def cache_dir() -> str:
    return _CACHE_DIR


def workspace_root() -> str:
    """The workspace the servers were started against, resolved, read at call time.

    ``_WORKSPACE_ROOT`` above is frozen at import, which is what the store's own layout
    wants; this is the live answer, which is what anything comparing paths against the
    tree wants. Two spellings of one file must not read as two files, hence realpath —
    ``build._declared_root`` deliberately does not resolve, for the opposite reason.
    """
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
    """The pointer naming the proxy a nameless ``proxy_eval`` op acts on.

    One file per MIMIR session. The store itself is shared — references, scaffolds and
    sealed fields are build artefacts of the project, and locking them is what
    ``_registry_lock`` is for — but *which optimisation a conversation is currently
    driving* is the conversation's own. With a single pointer, a ``proxy_eval(op='init')``
    in one session retargeted every nameless op in all the others, so a run meant for
    ``foo`` was measured against ``bar``.

    Unsuffixed when there is no session (the CLI, standalone runs, tests), which is also
    the name it has always had.
    """
    sid = session_id if session_id is not None else _mimir_session_id()
    name = f"active_session.{sid}" if sid else "active_session"
    return os.path.join(opt_runs_dir(), name)


def _mimir_session_id() -> str:
    """The MIMIR session this process is acting for, or "".

    Best-effort: a store reachable without the shared state helpers (a bare test
    fixture) must keep working, and "no session" is the pre-existing behaviour.
    """
    try:
        from state_paths import active_session_id
        return active_session_id()
    except Exception:
        return ""


# ── generic atomic IO ─────────────────────────────────────────────────────────

def _read_json(path: str, default=None):
    """Best-effort JSON read; returns *default* on a missing or corrupt file."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def _write_text_atomic(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _atomic_symlink(link: str, target: str) -> None:
    tmp = link + ".tmp"
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, link)


def _run_dir_names(parent: str) -> list[str]:
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
def _file_lock(lock_path: str):
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
def _registry_lock(create: bool = False):
    """Exclusive, re-entrant flock around registry read-modify-write operations.

    Re-entrant because mutation paths hold the lock while calling
    ``_load_registry``, which locks for its own read; the depth is thread-local
    so concurrent tool dispatch on worker threads still serializes.

    ``create`` says whether taking the lock may bring the store into existence.
    It defaults to False because the lock is on the read path too, and a plain
    ``proxy_get`` on a project that has never registered a proxy must not leave a
    ``proxy_bench/`` directory and a lock file behind in the user's tree — that is a
    read tool writing, including in plan mode where nothing should be written at
    all. With no store on disk there is no registry to race over, so the lock is skipped
    entirely and the read returns empty. Only :func:`registry.register` passes
    ``create=True``: registering a proxy is the act that creates the store.
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


def _load_registry() -> dict:
    # Always acquire the registry lock for all registry reads (even read-only)
    with _registry_lock():
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


def _load_registry_or_err() -> tuple[dict | None, str | None]:
    """Return (registry, None) on success or (None, error_message) on corruption."""
    try:
        return _load_registry(), None
    except RuntimeError as exc:
        return None, str(exc)


def _save_registry(reg: dict) -> None:
    # Always acquire the registry lock for all registry writes. No ``create``: a save
    # is always reached either under an outer lock that already took it (register) or
    # on an entry that was found, which means the store is already there. The atomic
    # write creates the directory itself in any case.
    with _registry_lock():
        _write_json_atomic(registry_path(), reg)


# ── run / reference layout ────────────────────────────────────────────────────

def _proxy_runs_dir(proxy_name: str) -> str:
    return os.path.join(runs_dir(), proxy_name)


def _active_link(proxy_name: str) -> str:
    return os.path.join(_proxy_runs_dir(proxy_name), "active")


def _ref_dir(reference_name: str) -> str:
    return os.path.join(refs_dir(), reference_name)


def _load_ref_metrics(reference_name: str) -> dict | None:
    return _read_json(os.path.join(_ref_dir(reference_name), "metrics.json"))


def _ref_output_path(reference_name: str) -> str | None:
    rd = _ref_dir(reference_name)
    for ext in (".npz", ".dat"):
        p = os.path.join(rd, f"output{ext}")
        if os.path.isfile(p):
            return p
    return None


# ── suites ────────────────────────────────────────────────────────────────────

def _suite_path(name: str) -> str:
    return os.path.join(suites_dir(), name, "suite.json")


def _load_suite(name: str) -> dict | None:
    return _read_json(_suite_path(name))


def _save_suite(name: str, suite: dict) -> None:
    _write_json_atomic(_suite_path(name), suite)


def _suite_results_dir(name: str) -> str:
    return os.path.join(suites_dir(), name, "results")


def _latest_suite_results(name: str) -> str | None:
    rd = _suite_results_dir(name)
    if not os.path.isdir(rd):
        return None
    entries = sorted(
        [d for d in os.listdir(rd) if os.path.isdir(os.path.join(rd, d))],
        reverse=True,
    )
    return os.path.join(rd, entries[0]) if entries else None


# ── optimization-session layout ───────────────────────────────────────────────

def _opt_session_runs_dir(proxy_name: str) -> str:
    """Directory that holds timestamped run dirs for one proxy's opt sessions."""
    return os.path.join(opt_runs_dir(), proxy_name)


def _opt_config_file(proxy_name: str) -> str:
    """Per-proxy opt config path: opt_runs/<proxy_name>/opt_config.json."""
    return os.path.join(_opt_session_runs_dir(proxy_name), "opt_config.json")


def _opt_active_link(proxy_name: str) -> str:
    return os.path.join(_opt_session_runs_dir(proxy_name), "active")


def _opt_ledger_file(proxy_name: str) -> str:
    """Append-only ledger of every completed run: opt_runs/<proxy>/ledger.jsonl."""
    return os.path.join(_opt_session_runs_dir(proxy_name), "ledger.jsonl")


def _opt_best_file(proxy_name: str) -> str:
    """Best-so-far pointer: opt_runs/<proxy>/best.json."""
    return os.path.join(_opt_session_runs_dir(proxy_name), "best.json")


def opt_git_dir() -> str:
    """Shadow repository for the tracked tree, inside the proxy store."""
    return os.path.join(cache_dir(), "opt.git")


def _load_opt_config(proxy_name: str = "") -> dict:
    name = _resolve_proxy_name(proxy_name)
    if not name:
        return {}
    return _read_json(_opt_config_file(name), {}) or {}


def _save_opt_config(cfg: dict, proxy_name: str = "") -> None:
    name = proxy_name or cfg.get("proxy_name", "")
    if not name:
        return
    _write_json_atomic(_opt_config_file(name), cfg)


def _entry_or_err(proxy_name: str) -> tuple[dict | None, dict | None]:
    """A registered proxy's entry, or the ready-to-return error saying why not.

    The same four lines stood in five ops with the same wording; a sixth caller
    would have copied them again.
    """
    reg, reg_err = _load_registry_or_err()
    if reg_err:
        return None, err(reg_err)
    if proxy_name not in reg:
        return None, err(f"Proxy '{proxy_name}' not registered.",
                         hint="Call proxy_manage(op='register', ...) first.")
    return reg[proxy_name], None


def _resolve_proxy_name(arg: str, session_id: str | None = None) -> str | None:
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


def _write_active_session(proxy_name: str, session_id: str | None = None) -> None:
    """Persist *proxy_name* as this MIMIR session's most-recently initialized proxy."""
    _write_text_atomic(active_session_file(session_id), proxy_name)


def _clear_active_session(session_id: str | None = None) -> None:
    """Drop the active-session pointer (best-effort). Leaves opt_config/runs intact
    for history; only the "current session" marker is removed, so nameless ops no
    longer resolve to it and the client's proxy-exec guard lifts."""
    try:
        os.remove(active_session_file(session_id))
    except FileNotFoundError:
        pass
    except OSError:
        pass

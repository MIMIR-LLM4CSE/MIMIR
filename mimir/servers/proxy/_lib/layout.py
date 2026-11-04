"""Where the proxy server keeps its state (``<workspace>/proxy_bench/``): the layout and its
plain reads, under one root the tests redirect (``_ROOT``). ``store`` persists over it.
"""
from __future__ import annotations

import json
import os

from ....shared.state_paths import active_session_id

# Under the workspace, so the experiment goes with the project; not `.mimir/` (the
# extensions directory) nor the state dir (references are megabytes). Workspace from
# MCP_FILES_ROOT, else cwd; MIMIR_PROXY_BENCH_DIR wins (tests).
_ROOT = (os.environ.get("MIMIR_PROXY_BENCH_DIR")
         or os.path.join(os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd()),
                         "proxy_bench"))


def cache_dir() -> str:
    return _ROOT


def registry_path() -> str:
    return os.path.join(_ROOT, "registry.json")


def requests_file() -> str:
    """The R&D requests put from the code map, across every proxy: one list."""
    return os.path.join(_ROOT, "requests.json")


def opt_runs_dir() -> str:
    return os.path.join(_ROOT, "opt_runs")


def history_dir() -> str:
    """The sessions the code map keeps after their state went (a clean, a re-baseline):
    ``history/<proxy>@<stamp>.json``."""
    return os.path.join(_ROOT, "history")


def opt_session_runs_dir(proxy_name: str) -> str:
    """Directory that holds timestamped run dirs for one proxy's opt sessions."""
    return os.path.join(opt_runs_dir(), proxy_name)


def opt_config_file(proxy_name: str) -> str:
    """Per-proxy opt config path: opt_runs/<proxy_name>/opt_config.json."""
    return os.path.join(opt_session_runs_dir(proxy_name), "opt_config.json")


def merge_file(proxy_name: str) -> str:
    """A merge session's state (its sources, its phase, what it kept): opt_runs/<merge>/merge.json."""
    return os.path.join(opt_session_runs_dir(proxy_name), "merge.json")


def active_sessions_dir() -> str:
    """Where the sessions' pointers live: ``active_sessions/<sid>/proxy``."""
    return os.path.join(opt_runs_dir(), "active_sessions")


def active_session_file(session_id: str | None = None) -> str:
    """The pointer to the proxy a nameless ``proxy_eval`` op acts on: one per MIMIR session.

    A fixed leaf under the session's id, so a sub-session's (``<sid>/subagents/<child>``)
    sits beside its parent's pointer, never in its place.
    """
    sid = session_id if session_id is not None else active_session_id()
    return os.path.join(active_sessions_dir(), *([sid] if sid else []), "proxy")


def read_json(path: str, default=None):
    """Best-effort JSON read; returns *default* on a missing or corrupt file."""
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return default


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
            with open(path) as fh:
                name = fh.read().strip()
            if name:
                return name
        except OSError:
            pass
    return None


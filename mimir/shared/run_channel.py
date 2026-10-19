"""The user's channel into a blocking run: two files under ``<session_state_dir>/runs/<channel>/``.

``current.json``, written by the server while it waits (job key, phase); ``divert``,
written by the client, which the wait loop consumes to leave the run going. *channel* is
the blocking tool's name. Both ends are here: a server resolves its directory from its
environment, the client names the state dir and session. Fail-open.
"""

import json
import os
import time

from . import state_paths
from .atomic import write_text
from .processes import is_running, proc_starttime

# How long a divert request stays meaningful. Past this it names a run that has
# almost certainly ended, and consuming it would detach the *next* command instead.
_STALE_S = 60.0


def _dir(channel: str, base: str | None = None, session_id: str | None = None) -> str:
    return os.path.join(state_paths.session_state_dir(base, session_id), "runs", channel)


def _read_current(channel: str, base: str | None = None,
                  session_id: str | None = None) -> dict | None:
    """The announcement as it stands, or None when there is none to read."""
    try:
        with open(os.path.join(_dir(channel, base, session_id), "current.json"),
                  encoding="utf-8") as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return None
    return run if isinstance(run, dict) and run.get("job_key") else None


# ── The server's end ──────────────────────────────────────────────────────────

def publish(channel: str, job_key: str, pid: int, command: str, cwd: str,
            deadline_s: float, phase: str = "", percent: float | None = None) -> None:
    """Announce the foreground run the client may divert, clearing a leftover divert request.

    ``server_pid`` and ``server_started`` let the client tell a live announcement from
    a crashed server's, even once its pid is reused.
    """
    try:
        base = _dir(channel)
        os.makedirs(base, exist_ok=True)
        stale = os.path.join(base, "divert")
        try:
            if os.path.exists(stale):
                os.unlink(stale)
        except OSError:
            pass
        write_text(os.path.join(base, "current.json"), json.dumps({
            "job_key": job_key,
            "pid": pid,
            "server_pid": os.getpid(),
            "server_started": proc_starttime(os.getpid()),
            "command": command,
            "cwd": cwd,
            "deadline_s": deadline_s,
            "started_at": time.time(),
            "phase": phase,
            "percent": percent,
        }))
    except OSError:
        pass


def update(channel: str, job_key: str, phase: str | None = None,
           percent: float | None = None) -> None:
    """Refresh what the announced run is doing, if it is still *job_key* (never touches ``divert``)."""
    run = _read_current(channel)
    if run is None or run.get("job_key") != job_key:
        return
    if phase is not None:
        run["phase"] = phase
    run["percent"] = percent
    try:
        write_text(os.path.join(_dir(channel), "current.json"), json.dumps(run))
    except OSError:
        pass


def clear(channel: str, job_key: str) -> None:
    """Retract the announcement, but only if it still names *job_key*."""
    try:
        base = _dir(channel)
        path = os.path.join(base, "current.json")
        with open(path, encoding="utf-8") as fh:
            if json.load(fh).get("job_key") != job_key:
                return
        os.unlink(path)
        try:
            os.unlink(os.path.join(base, "divert"))
        except OSError:
            pass
    except (OSError, ValueError):
        pass


def requested(channel: str, job_key: str) -> bool:
    """True — and consume the request — iff a live divert names *job_key*."""
    path = os.path.join(_dir(channel), "divert")
    try:
        if time.time() - os.path.getmtime(path) > _STALE_S:
            os.unlink(path)
            return False
        with open(path, encoding="utf-8") as fh:
            named = fh.read().strip()
    except (OSError, ValueError):
        return False
    if named != job_key:
        return False
    try:
        os.unlink(path)
    except OSError:
        pass
    return True


# ── The client's end ──────────────────────────────────────────────────────────

def _server_alive(run: dict) -> bool:
    """Whether the announcing server still runs; an announcement without its pid is believed."""
    pid = run.get("server_pid")
    if not isinstance(pid, int) or pid <= 0:
        return True
    started = run.get("server_started")
    return is_running(pid, started if isinstance(started, int) else None)


def current_run(channel: str, base: str | None = None,
                session_id: str | None = None) -> dict | None:
    """The foreground run a live server is waiting on for *channel*, if there is one."""
    run = _read_current(channel, base, session_id)
    return run if run is not None and _server_alive(run) else None


def request_divert(channel: str, base: str | None = None,
                   session_id: str | None = None) -> dict | None:
    """Ask the server to detach the current run; return what was targeted.

    ``None`` means there was nothing to divert — almost always because the command
    finished between the click and the read. The caller says so rather than
    pretending something happened.
    """
    run = current_run(channel, base, session_id)
    if run is None:
        return None
    try:
        write_text(os.path.join(_dir(channel, base, session_id), "divert"), str(run["job_key"]))
    except OSError:
        return None
    return run

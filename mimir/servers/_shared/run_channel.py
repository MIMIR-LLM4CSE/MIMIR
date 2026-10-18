"""The user's channel into a blocking run: two files under ``<session_state_dir>/runs/<channel>/``.

``current.json``, written by the server while it waits (job key, phase); ``divert``,
written by the client, which the wait loop consumes to stop waiting and leave the run
going. *channel* is the blocking tool's name; one call per tool per session. Fail-open.
"""

import json
import os
import time

from .atomic import write_text

# How long a divert request stays meaningful. Past this it names a run that has
# almost certainly ended, and consuming it would detach the *next* command instead.
_STALE_S = 60.0


def _dir(channel: str) -> str:
    from .state_paths import session_state_dir
    return os.path.join(session_state_dir(), "runs", channel)


def _read_current(channel: str) -> dict | None:
    """The announcement as it stands, or None when there is none to read."""
    try:
        with open(os.path.join(_dir(channel), "current.json"), encoding="utf-8") as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return None
    return run if isinstance(run, dict) and run.get("job_key") else None


def publish(channel: str, job_key: str, pid: int, command: str, cwd: str,
            deadline_s: float, phase: str = "", percent: float | None = None) -> None:
    """Announce the foreground run the client may divert, clearing a leftover divert request.

    ``server_pid`` lets the client tell a live announcement from a crashed server's.
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

"""The user's channel into a run that is already going.

A tool whose body blocks cannot be reached through the model: the client's worker
thread is parked awaiting the tool result, and steering is only drained at a step
boundary, so an instruction routed that way would arrive after the run it meant to
divert had already ended. It cannot be reached through MCP either — a synchronous
tool body holds its server's event loop for the whole call, and even an async one
(``proxy_eval``) is a single awaited call with no second channel into it.

What is left is the shared state dir, which both ends can touch at any moment. The
server announces the run it is waiting on; the client names that run in a request
file the server's wait loop consumes on its next tick. The server also republishes
what the run is *doing*, which is what lets a blocking call show a live phase
instead of a mute spinner.

Two files, under ``<state_dir>/runs/<channel>/``:

``current.json``  written by the server the moment it starts waiting, so the client
                  can learn which job a blocking run is, without the key ever
                  becoming a tool parameter the model could set. Refreshed by
                  :func:`update` as the run moves through its phases.
``divert``        written by the client, naming that key. The waiting loop consumes
                  it and stops waiting, leaving the process alive.

*channel* is the name of the tool that blocks — ``bash_run``, ``proxy_eval``. It is
registry data on both ends: the server passes the name it registered, and the client
reads it off the tool row it is diverting. Neither end ever tests it against a
literal. One file each is unambiguous because at most one foreground call per tool
can be in flight at any instant — that is the same blocking property stated above,
read as a guarantee rather than a limitation.

Everything here is best-effort and fail-open: a missing, stale or corrupt sidecar
must leave the run behaving exactly as it did before this module existed.
"""

import json
import os
import time

# How long a divert request stays meaningful. Past this it names a run that has
# almost certainly ended, and consuming it would detach the *next* command instead.
_STALE_S = 60.0


def _dir(channel: str) -> str:
    from state_paths import state_dir
    return os.path.join(state_dir(), "runs", channel)


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


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
    """Announce the foreground run the client may divert.

    Also clears a divert left over from an earlier run: a request that arrived just
    after its target finished would otherwise detach the next command the agent runs,
    which is the one failure of this design a user could not explain.

    ``server_pid`` rides along so the client can tell a live announcement from one a
    crashed server left behind — see the client half's ``current_run``.
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
        _write_atomic(os.path.join(base, "current.json"), json.dumps({
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
    """Refresh what the announced run is doing, if it is still *job_key*.

    Deliberately does not touch the ``divert`` file. Clearing a leftover request is a
    start-of-run act (see :func:`publish`); this runs on every tick of a wait loop,
    while a request the user just made may be in flight, and consuming it here would
    drop the click on the floor.
    """
    run = _read_current(channel)
    if run is None or run.get("job_key") != job_key:
        return
    if phase is not None:
        run["phase"] = phase
    run["percent"] = percent
    try:
        _write_atomic(os.path.join(_dir(channel), "current.json"), json.dumps(run))
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

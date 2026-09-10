"""The user's channel into a shell run that is already going.

``bash_run`` is a synchronous MCP tool, so its body blocks the bash server's whole
event loop for the duration of the command: no second tool call and no MCP cancel
notification can be serviced while a foreground run is in flight. The server's
environment is frozen at spawn, so — exactly as with the approved-paths allowlist
(cf. ``_is_within_workspace``) — a file on the shared state dir is the only live
client→server channel left.

Two files, under ``<state_dir>/bash_run/``:

``current.json``  written by the server the moment it starts waiting, so the client
                  can learn which job a foreground run is, without the key ever
                  becoming a tool parameter the model could set.
``divert``        written by the client, naming that key. The waiting loop consumes
                  it and stops waiting, leaving the process alive.

One file each is unambiguous because at most one foreground ``bash_run`` can be in
flight per bash server at any instant — that is the same blocking property stated
above, read as a guarantee rather than a limitation.

Everything here is best-effort and fail-open: a missing, stale or corrupt sidecar
must leave the run behaving exactly as it did before this module existed.
"""

import json
import os
import time

# How long a divert request stays meaningful. Past this it names a run that has
# almost certainly ended, and consuming it would detach the *next* command instead.
_STALE_S = 60.0


def _dir() -> str:
    from state_paths import state_dir
    return os.path.join(state_dir(), "bash_run")


def _write_atomic(path: str, text: str) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def publish(job_key: str, pid: int, command: str, cwd: str, deadline_s: float) -> None:
    """Announce the foreground run the client may divert.

    Also clears a divert left over from an earlier run: a request that arrived just
    after its target finished would otherwise detach the next command the agent runs,
    which is the one failure of this design a user could not explain.
    """
    try:
        base = _dir()
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
            "command": command,
            "cwd": cwd,
            "deadline_s": deadline_s,
            "started_at": time.time(),
        }))
    except OSError:
        pass


def clear(job_key: str) -> None:
    """Retract the announcement, but only if it still names *job_key*."""
    try:
        base = _dir()
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


def requested(job_key: str) -> bool:
    """True — and consume the request — iff a live divert names *job_key*."""
    path = os.path.join(_dir(), "divert")
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

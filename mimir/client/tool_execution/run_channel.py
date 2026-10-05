"""The client half of the divert channel — see servers/_shared/run_channel.py.

A blocking run cannot be reached through the model: the worker thread is parked
awaiting the tool result, and steering is only drained at a step boundary, so an
instruction routed that way would arrive after the run it meant to divert had
already ended. It cannot be reached through MCP either, since the call itself is
what the server is busy with.

What is left is the shared state dir, which both ends can touch at any moment. The
server announces the run it is waiting on; this module names that run in a request
file the server's wait loop consumes on its next tick, and reads back the phase the
server republishes so a blocking call can show what it is doing.

*channel* is the name of the tool that blocks. It reaches here off the tool row
being diverted, which carries it as registry data — this module never tests it
against a literal, and the name is never shown to the user.

*session_id* says whose run is meant. Sessions run turns concurrently, so two of them
can each have a ``bash_run`` blocking, and the channel directory lives under each
session's own state dir; a divert addressed to no session in particular would detach
whichever command happened to announce itself there. It is passed in rather than read
from the environment because this half runs in the client, where every session shares
one ``os.environ``.

Two small files, one format, and the two ends of it are pinned together by
tests/test_run_channel.py.

Best-effort throughout: if anything here fails, the run simply carries on blocking, which
is what it does when nothing asks it to detach.
"""

import json
import os

from ..config.constants import STATE_DIR


def _dir(channel: str, session_id: str | None = None) -> str:
    from ...servers._shared.state_paths import session_state_dir
    return os.path.join(session_state_dir(STATE_DIR, session_id), "runs", channel)


def _server_alive(run: dict) -> bool:
    """Whether the server that made this announcement is still there.

    A server that crashed mid-run leaves ``current.json`` behind, and without this a
    divert would report success while nothing at all was listening. Both ends are on
    one machine — the state dir is local — so the process table is the honest answer.

    Fail-open, as everything here is: an announcement from a server that predates the
    ``server_pid`` field, or a platform with no ``/proc``, is accepted.
    """
    pid = run.get("server_pid")
    if not isinstance(pid, int) or pid <= 0:
        return True
    if not os.path.isdir("/proc"):
        return True
    return os.path.exists(f"/proc/{pid}")


def current_run(channel: str, session_id: str | None = None) -> dict | None:
    """The foreground run the server is waiting on for *channel*, if there is one."""
    try:
        with open(os.path.join(_dir(channel, session_id), "current.json"), encoding="utf-8") as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(run, dict) or not run.get("job_key"):
        return None
    return run if _server_alive(run) else None


def request_divert(channel: str, session_id: str | None = None) -> dict | None:
    """Ask the server to detach the current run; return what was targeted.

    ``None`` means there was nothing to divert — almost always because the command
    finished between the click and the read. The caller says so rather than
    pretending something happened.
    """
    run = current_run(channel, session_id)
    if run is None:
        return None
    try:
        base = _dir(channel, session_id)
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "divert")
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(str(run["job_key"]))
        os.replace(tmp, path)
    except OSError:
        return None
    return run

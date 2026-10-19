"""The client half of the divert channel (see shared/run_channel.py).

A blocking run is reachable through neither the model nor MCP, so the two ends meet in
the session's state dir: the server announces its run, this writes the divert request
and reads back the phase. *channel* is the blocking tool's name; best-effort throughout.
"""

import json
import os

from ..config import paths
from ...shared.atomic import write_text
from ...shared.state_paths import session_state_dir


def _dir(channel: str, session_id: str | None = None) -> str:
    return os.path.join(session_state_dir(paths.STATE_DIR, session_id), "runs", channel)


def _server_alive(run: dict) -> bool:
    """Whether the announcing server is still running (fail-open without ``server_pid`` or ``/proc``)."""
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
        write_text(os.path.join(_dir(channel, session_id), "divert"), str(run["job_key"]))
    except OSError:
        return None
    return run

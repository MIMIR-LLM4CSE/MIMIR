"""The client half of the divert channel — see servers/workspace/_bash_divert.py.

A shell run that blocks the turn cannot be reached through the model: the worker
thread is parked awaiting the tool result, and steering is only drained at a step
boundary, so an instruction routed that way would arrive after the run it meant to
divert had already ended. It cannot be reached through MCP either, since the
synchronous tool body holds the bash server's event loop for the whole command.

What is left is the shared state dir, which both ends can touch at any moment. The
server announces the run it is waiting on; this module names that run in a request
file the server's wait loop consumes on its next tick. Two small files, one format,
and the two ends of it are pinned together by tests/test_bash_divert.py.

Best-effort throughout: if anything here fails, the run simply carries on blocking,
which is what it did before this existed.
"""

import json
import os

from ..config.constants import STATE_DIR


def _dir() -> str:
    return os.path.join(STATE_DIR, "bash_run")


def current_run() -> dict | None:
    """The foreground shell run the server is waiting on, if there is one."""
    try:
        with open(os.path.join(_dir(), "current.json"), encoding="utf-8") as fh:
            run = json.load(fh)
    except (OSError, ValueError):
        return None
    return run if isinstance(run, dict) and run.get("job_key") else None


def request_divert() -> dict | None:
    """Ask the server to detach the current run; return what was targeted.

    ``None`` means there was nothing to divert — almost always because the command
    finished between the click and the read. The caller says so rather than
    pretending something happened.
    """
    run = current_run()
    if run is None:
        return None
    try:
        base = _dir()
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "divert")
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(str(run["job_key"]))
        os.replace(tmp, path)
    except OSError:
        return None
    return run

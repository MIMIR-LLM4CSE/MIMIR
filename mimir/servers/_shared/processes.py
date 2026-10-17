"""Whether a recorded pid is still the process that was recorded.

A job directory outlives its process, and pids are recycled, so a run is identified by
its pid *and* its start time (field 22 of ``/proc/<pid>/stat``).
"""
from __future__ import annotations

import os


def _stat_fields(pid: int) -> list[str] | None:
    """The fields of ``/proc/<pid>/stat`` after the command name, or None."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
    except OSError:
        return None
    # The command name is parenthesized and may contain spaces: split after its close.
    idx = data.rfind(")")
    return data[idx + 2:].split() if idx >= 0 else None


def proc_starttime(pid: int) -> int | None:
    """The clock tick *pid* started at, or None when it cannot be read."""
    fields = _stat_fields(pid)
    try:
        return int(fields[19]) if fields else None
    except (IndexError, ValueError):
        return None


def is_running(pid: int, expected_starttime: int | None = None) -> bool:
    """Whether *pid* exists, is not a zombie, and started at *expected_starttime*.

    A zombie counts as finished: nothing reaps a detached child, so its table entry
    lingers after it exits.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    fields = _stat_fields(pid)
    if fields and fields[0] == "Z":
        return False
    if expected_starttime is not None:
        actual = proc_starttime(pid)
        if actual is not None and actual != expected_starttime:
            return False
    return True

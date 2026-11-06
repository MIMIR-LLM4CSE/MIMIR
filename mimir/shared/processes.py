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


def is_running(pid: object, expected_starttime: int | None = None) -> bool:
    """Whether *pid* exists, is not a zombie, and started at *expected_starttime*.

    A zombie counts as finished: nothing reaps a detached child, so its table entry
    lingers after it exits.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, OverflowError):  # a pid past the platform's range is no process
        return False
    fields = _stat_fields(pid)
    if fields and fields[0] == "Z":
        return False
    if expected_starttime is not None:
        actual = proc_starttime(pid)
        if actual is not None and actual != expected_starttime:
            return False
    return True


def session_argvs(session: int) -> list[list[str]]:
    """The argv of every process of *session* (its leader's pid); [] without ``/proc``."""
    try:
        pids = [int(name) for name in os.listdir("/proc") if name.isdigit()]
    except OSError:
        return []
    out = []
    for pid in pids:
        fields = _stat_fields(pid)
        if not fields or len(fields) < 4 or fields[3] != str(session):
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                raw = fh.read()
        except OSError:
            continue
        argv = [word.decode(errors="replace") for word in raw.split(b"\0")[:-1]]
        if argv:
            out.append(argv)
    return out

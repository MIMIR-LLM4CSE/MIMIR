"""Run a subprocess under a timeout that kills its whole process group.

``subprocess.run`` kills only its child; what that child forked (bash forks for a
compound script) survives and corrupts later measurements. The child gets its own
session; SIGTERM, then SIGKILL after a grace.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

# How long a process group gets to exit on SIGTERM before SIGKILL. Short: this runs
# after a timeout the caller has already waited out.
_GRACE_S = 2.0


def _terminate_group(proc: subprocess.Popen) -> None:
    """Signal the child's whole process group, escalating TERM -> KILL."""
    if not hasattr(os, "killpg"):  # pragma: no cover - not POSIX
        proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return  # already reaped; nothing to signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return  # group is gone
        try:
            proc.wait(timeout=_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def run(argv: list[str], *, timeout: float | None = None, **kwargs: Any):
    """:func:`subprocess.run`, but a timeout kills the child's process group.

    ``start_new_session`` is forced; ``stdin`` defaults to ``DEVNULL``, since a server's stdin
    is its MCP pipe and a reading child would eat the protocol.
    """
    kwargs.setdefault("start_new_session", True)
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    with subprocess.Popen(argv, **kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_group(proc)
            # The group is dead, so the pipes are closed and this cannot block. Bounded
            # anyway: draining output must never be a second way to hang on a timeout.
            try:
                stdout, stderr = proc.communicate(timeout=_GRACE_S)
            except subprocess.TimeoutExpired:
                stdout = stderr = None
            raise subprocess.TimeoutExpired(argv, timeout or 0, output=stdout, stderr=stderr)
        except BaseException:
            # Cancelled or interrupted: the same leak, by a different route.
            _terminate_group(proc)
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)

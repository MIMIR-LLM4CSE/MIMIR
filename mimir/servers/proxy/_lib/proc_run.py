"""Run a subprocess under a timeout that kills its whole process group.

``subprocess.run`` kills only its child; what that child forked (bash forks for a
compound script) survives and corrupts later measurements. The child gets its own
session; SIGTERM, then SIGKILL after a grace. A measured run also reports its memory:
the largest process's peak, from ``wait4``, and the sum of every process's own peak (an
MPI run's ranks), sampled from ``/proc`` while it runs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
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


def run(argv: list[str], *, timeout: float | None = None, rusage: bool = False, **kwargs: Any):
    """:func:`subprocess.run`, but a timeout kills the child's process group.

    ``start_new_session`` is forced; ``stdin`` defaults to ``DEVNULL``, since a server's stdin
    is its MCP pipe and a reading child would eat the protocol. With *rusage* (output to
    files, not pipes), the result's ``peak_rss_kb`` is the tree's largest resident set.
    """
    kwargs.setdefault("start_new_session", True)
    kwargs.setdefault("stdin", subprocess.DEVNULL)
    if rusage and hasattr(os, "wait4"):
        return _run_measured(argv, timeout, kwargs)
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


def _run_measured(argv: list[str], timeout: float | None, kwargs: dict):
    """:func:`run` reaping the child with ``wait4``, whose ``ru_maxrss`` covers the child
    and every descendant it reaped: the largest process of the tree (``peak_rss_kb``);
    and the sum of each process's own peak (``peak_rss_total_kb``, over ``processes``)."""
    if subprocess.PIPE in (kwargs.get("stdout"), kwargs.get("stderr")):
        raise ValueError("a measured run writes its output to files: a pipe would need draining")
    reaped: dict = {}
    spawner = _own_peak_kb()
    with subprocess.Popen(argv, **kwargs) as proc:
        def reap() -> None:
            _pid, status, usage = os.wait4(proc.pid, 0)
            reaped.update(status=status, usage=usage)
        waiter = threading.Thread(target=reap, daemon=True)
        waiter.start()
        # The child leads its own session, so its session id is its pid: every process
        # the run starts, MPI ranks included, carries it unless it leaves on purpose.
        tree = SessionMemory(proc.pid)
        tree.start()
        try:
            waiter.join(timeout)
            timed_out = waiter.is_alive()
        except BaseException:
            timed_out = True
            raise
        finally:
            tree.stop()
            if timed_out:
                _end_group(proc.pid, waiter)
            # Reaped here, so Popen must not wait for it again.
            proc.returncode = os.waitstatus_to_exitcode(reaped["status"]) if reaped else -signal.SIGKILL
        if timed_out:
            raise subprocess.TimeoutExpired(argv, timeout or 0)
        done: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(argv, proc.returncode, None, None)
        maxrss = reaped["usage"].ru_maxrss
        maxrss = maxrss // 1024 if sys.platform == "darwin" else maxrss
        sampled = max(tree.peaks.values(), default=0)
        # Linux carries the spawner's own peak into the child's ru_maxrss across exec, so
        # wait4's figure is the child's only above it; below, the samples are.
        done.peak_rss_kb = max(maxrss, sampled) if maxrss > spawner or not sampled else sampled  # type: ignore[attr-defined]
        if tree.peaks:
            done.peak_rss_total_kb = max(sum(tree.peaks.values()), done.peak_rss_kb)  # type: ignore[attr-defined]
            done.processes = len(tree.peaks)  # type: ignore[attr-defined]
        return done


def _own_peak_kb() -> int:
    """This process's own peak resident set, in KB (0 where /proc does not say)."""
    try:
        with open("/proc/self/status") as fh:
            return next((int(line.split()[1]) for line in fh if line.startswith("VmHWM:")), 0)
    except (OSError, ValueError):
        return 0


class SessionMemory:
    """Each process of a session's own peak resident set (``VmHWM``), sampled from
    ``/proc`` while the session runs; nothing where there is no ``/proc``."""

    INTERVAL_S = 0.1

    def __init__(self, sid: int) -> None:
        self.sid = sid
        self.peaks: dict[tuple[str, bytes], int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if os.path.isdir("/proc"):
            self._thread = threading.Thread(target=self._watch, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(1)

    def _watch(self) -> None:
        while True:
            self.sample()
            if self._stop.wait(self.INTERVAL_S):
                return

    def sample(self) -> None:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as fh:
                    stat = fh.read()
                # After the command's ")": state, ppid, pgrp, session, ..., starttime (20th).
                fields = stat[stat.rindex(b")") + 2:].split()
                if int(fields[3]) != self.sid:
                    continue
                with open(f"/proc/{name}/status") as fh:
                    hwm = next((int(line.split()[1]) for line in fh if line.startswith("VmHWM:")), 0)
            except (OSError, ValueError, IndexError):
                continue                # gone between the listing and the read
            key = (name, fields[19])    # a recycled pid is another process
            if hwm > self.peaks.get(key, 0):
                self.peaks[key] = hwm


def _end_group(pid: int, waiter: threading.Thread) -> None:
    """TERM then KILL the group of *pid*, leaving the reaping to *waiter*."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not waiter.is_alive():
            return
        try:
            os.killpg(pgid, sig)
        except OSError:
            return
        waiter.join(_GRACE_S)

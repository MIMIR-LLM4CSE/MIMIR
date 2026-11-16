"""Run a subprocess under a timeout that kills its whole process group.

``subprocess.run`` kills only its child; what that child forked survives and corrupts
later measurements, so the child gets its own session: SIGTERM, then SIGKILL. A measured
run also reports its memory (the largest process's peak and every process's summed), what
its processes run (``fingerprint``) and its measured regions (``region``).
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from .region import Regions

# What a measured run leaves beside its output where no MIMIR process waits for it.
REGION_FILE, PROCESSES_FILE = "region.json", "processes.json"

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


def run(argv: list[str], *, timeout: float | None = None, rusage: bool = False,
        region: Callable[[str, list[str]], None] | None = None, **kwargs: Any):
    """:func:`subprocess.run`, but a timeout kills the child's process group; ``stdin`` is
    ``DEVNULL`` (a server's is its MCP pipe) unless ``input`` is given. With *rusage*
    (output to files): ``peak_rss_kb``, ``region``, ``fingerprint``. *region* is told
    ``"begin"``/``"end"`` and the run's pids as the run's regions open and close.
    """
    given = kwargs.pop("input", None)
    kwargs.setdefault("start_new_session", True)
    kwargs.setdefault("stdin", subprocess.PIPE if given is not None else subprocess.DEVNULL)
    if (rusage or region) and hasattr(os, "wait4"):
        return _run_measured(argv, timeout, kwargs, region)
    with subprocess.Popen(argv, **kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(given, timeout=timeout)
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


def _run_measured(argv: list[str], timeout: float | None, kwargs: dict,
                  on_mark: Callable[[str, list[str]], None] | None = None):
    """:func:`run` reaping the child with ``wait4``, whose ``ru_maxrss`` covers the child
    and every descendant it reaped: the largest process of the tree (``peak_rss_kb``);
    and the sum of each process's own peak (``peak_rss_total_kb``, over ``processes``).
    The run marks its regions on the FIFO ``MIMIR_REGION`` names (``region``)."""
    if subprocess.PIPE in (kwargs.get("stdout"), kwargs.get("stderr")):
        raise ValueError("a measured run writes its output to files: a pipe would need draining")
    reaped: dict = {}
    spawner = _own_peak_kb()
    tree = SessionMemory(0)
    regions = Regions(tree.pids, tree.note, on_mark)
    marked = regions.env()
    if marked:
        kwargs["env"] = {**(kwargs.get("env") or os.environ), **marked}
    try:
        done = _reap(argv, timeout, kwargs, tree, reaped, spawner)
    finally:
        regions.close()
    done.region = regions.summary()  # type: ignore[attr-defined]
    return done


def _reap(argv: list[str], timeout: float | None, kwargs: dict, tree: "SessionMemory",
          reaped: dict, spawner: int):
    with subprocess.Popen(argv, **kwargs) as proc:
        def reap() -> None:
            _pid, status, usage = os.wait4(proc.pid, 0)
            reaped.update(status=status, usage=usage)
        waiter = threading.Thread(target=reap, daemon=True)
        waiter.start()
        # The child leads its own session, so its session id is its pid: every process
        # the run starts, MPI ranks included, carries it unless it leaves on purpose.
        tree.sid = proc.pid
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
        done.fingerprint = tree.fingerprint()  # type: ignore[attr-defined]
        return done


def _own_peak_kb() -> int:
    """This process's own peak resident set, in KB (0 where /proc does not say)."""
    try:
        with open("/proc/self/status") as fh:
            return next((int(line.split()[1]) for line in fh if line.startswith("VmHWM:")), 0)
    except (OSError, ValueError):
        return 0


# The runtimes a process can load that say what runs: an interpreter, a language's
# runtime, a parallel or a GPU one, a math library.
_RUNTIME_RE = re.compile(r"/lib(python\d[\d.]*|gfortran|ifcore|ifport|flang\w*|stdc\+\+|c\+\+|gomp|iomp5|omp|"
                         r"mpi\w*|cudart|cuda|nvrtc|amdhip64|hiprtc|tbb|openblas|mkl_core|julia)[.\-][^/]*$")
_MAPS_EVERY_S = 1.0


class SessionMemory:
    """Each process of a session's own peak resident set (``VmHWM``), sampled from
    ``/proc`` while the session runs, and what each runs (its program and the runtimes
    it loads, ``fingerprint``); nothing where there is no ``/proc``."""

    INTERVAL_S = 0.1

    def __init__(self, sid: int) -> None:
        self.sid = sid
        self.peaks: dict[tuple[str, bytes], int] = {}
        self.programs: dict[tuple[str, bytes], str] = {}
        self.runtimes: set[str] = set()
        self._maps_read: dict[tuple[str, bytes], float] = {}
        self._lock = threading.Lock()
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

    def _members(self) -> list[tuple[str, bytes]]:
        """The session's processes now, each as (pid, start time): a recycled pid is
        another process."""
        found = []
        if not self.sid or not os.path.isdir("/proc"):
            return found
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            try:
                with open(f"/proc/{name}/stat", "rb") as fh:
                    stat = fh.read()
                # After the command's ")": state, ppid, pgrp, session, ..., starttime (20th).
                fields = stat[stat.rindex(b")") + 2:].split()
                if int(fields[3]) == self.sid:
                    found.append((name, fields[19]))
            except (OSError, ValueError, IndexError):
                continue                # gone between the listing and the read
        return found

    def pids(self) -> list[str]:
        return [pid for pid, _ in self._members()]

    def note(self, pid: str, hwm: int) -> None:
        """Keep *hwm* as *pid*'s peak if it is above the one known (before a region
        restarts the mark, which would hide it from the next sample)."""
        key = next((k for k in self._members() if k[0] == pid), None)
        if key:
            with self._lock:
                if hwm > self.peaks.get(key, 0):
                    self.peaks[key] = hwm

    def sample(self) -> None:
        now = time.monotonic()
        for key in self._members():
            name = key[0]
            try:
                with open(f"/proc/{name}/status") as fh:
                    hwm = next((int(line.split()[1]) for line in fh if line.startswith("VmHWM:")), 0)
            except (OSError, ValueError, IndexError):
                continue
            with self._lock:
                if hwm > self.peaks.get(key, 0):
                    self.peaks[key] = hwm
            if now - self._maps_read.get(key, -_MAPS_EVERY_S) >= _MAPS_EVERY_S:
                # Read again later: an exec changes the program, an import loads a runtime.
                self._maps_read[key] = now
                self._what_runs(key)

    def _what_runs(self, key: tuple[str, bytes]) -> None:
        try:
            self.programs[key] = os.path.basename(os.readlink(f"/proc/{key[0]}/exe"))
            with open(f"/proc/{key[0]}/maps") as fh:
                for line in fh:
                    m = _RUNTIME_RE.search(line.rstrip())
                    if m:
                        self.runtimes.add(m.group(1))
        except OSError:
            pass

    def fingerprint(self) -> dict:
        """What the session ran: its programs and the runtimes they loaded."""
        return {"programs": sorted(set(self.programs.values())), "runtimes": sorted(self.runtimes),
                "processes": len(self.peaks)}


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

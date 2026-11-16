"""The measured region of a run, the harness's call to the code, timed and weighed by
the server so the harness's own work counts for nothing. The run writes ``begin`` and
``end`` lines to the FIFO ``MIMIR_REGION`` names and waits for a byte on ``<path>.ack``;
the server reads each process's resident set and restarts its peak (``clear_refs``) at
``begin``, reads the peak at ``end``. Regions nest (each MPI rank may mark); in turn,
their times add. Helpers for Python, C, C++ and Fortran: ``servers/proxy/region``.
"""
from __future__ import annotations

import os
import select
import shutil
import tempfile
import threading
import time
from typing import Callable

ENV, ACK = "MIMIR_REGION", ".ack"


def _status(pid: str) -> dict[str, int]:
    """A process's VmRSS and VmHWM, in KB; empty once it is gone."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            return {k: int(v.split()[0]) for k, _, v in (line.partition(":") for line in fh)
                    if k in ("VmRSS", "VmHWM")}
    except (OSError, ValueError, IndexError):
        return {}


def _clear_peak(pid: str) -> bool:
    """Restart a process's high-water mark at its resident set now (Linux >= 4.0)."""
    try:
        with open(f"/proc/{pid}/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


class Regions:
    """The FIFO a run marks its regions on, and what the server read at each mark.
    *processes* gives the run's pids; *before_reset* is told each one's peak before it
    is restarted, so the run's own peak survives; *on_mark* is told the mark and the
    pids (a profiler's switch). Without ``/proc`` or a FIFO, ``env()`` is empty."""

    def __init__(self, processes: Callable[[], list[str]] | None = None,
                 before_reset: Callable[[str, int], None] | None = None,
                 on_mark: Callable[[str, list[str]], None] | None = None) -> None:
        self._processes = processes or (lambda: [])
        self._before_reset = before_reset
        self._on_mark = on_mark
        self._dir = ""
        self._fd = self._ack = -1
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._depth = 0
        self._opened = 0.0
        self._base: dict[str, int] = {}
        self.time_s = 0.0
        self.count = 0
        self.peak_kb = self.peak_total_kb = self.extra_kb = 0
        self.weighed = True

    def env(self) -> dict[str, str]:
        """``MIMIR_REGION`` for the run, the FIFO open and read from now on; {} when no
        FIFO can be made here."""
        if not hasattr(os, "mkfifo") or not os.path.isdir("/proc"):
            return {}
        try:
            # Node-local: a FIFO on a shared filesystem is local to each client, and a
            # rank on another node must find nothing there rather than wait on it.
            base = "/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None
            self._dir = tempfile.mkdtemp(prefix="mimir-region-", dir=base)
            path = os.path.join(self._dir, "region")
            os.mkfifo(path, 0o600)
            os.mkfifo(path + ACK, 0o600)
            # Read-write ends never block on open and never read end-of-file between writers.
            self._fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
            self._ack = os.open(path + ACK, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            self.close()
            return {}
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return {ENV: path}

    def close(self) -> None:
        """Stop reading, close the region still open at exit, remove the FIFOs."""
        self._stop.set()
        if self._thread:
            self._thread.join(2)
        if self._depth:
            self._depth = 1
            self._mark("end", time.monotonic())
        for fd in (self._fd, self._ack):
            if fd >= 0:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._fd = self._ack = -1
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = ""

    def summary(self) -> dict | None:
        """What the regions measured, or None for a run that marked none."""
        if not self.count:
            return None
        return {"time_s": round(self.time_s, 6), "count": self.count,
                **({"peak_kb": self.peak_kb, "peak_total_kb": self.peak_total_kb,
                    "extra_kb": self.extra_kb} if self.weighed and self.peak_kb else {})}

    # ── reading the marks ────────────────────────────────────────────────────

    def _read(self) -> None:
        pending = b""
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.1)
                if not ready:
                    continue
                chunk = os.read(self._fd, 4096)
            except (OSError, ValueError):
                return
            now = time.monotonic()
            pending += chunk
            *lines, pending = pending.split(b"\n")
            for line in lines:
                word = line.strip().split(b" ", 1)[0].decode("ascii", "replace")
                if word in ("begin", "end"):
                    self._mark(word, now)
                    try:
                        os.write(self._ack, b"\n")
                    except OSError:
                        pass

    def _mark(self, word: str, now: float) -> None:
        if word == "begin":
            self._depth += 1
            if self._depth == 1:
                self._open()
        elif self._depth:
            self._depth -= 1
            if self._depth == 0:
                self._shut(now)

    def _open(self) -> None:
        self._base = {}
        pids = self._processes()
        for pid in pids:
            st = _status(pid)
            if not st:
                continue
            if self._before_reset and st.get("VmHWM"):
                self._before_reset(pid, st["VmHWM"])
            self._base[pid] = st.get("VmRSS", 0)
            if not _clear_peak(pid):
                self.weighed = False
        if self._on_mark:
            self._on_mark("begin", pids)
        # From the answer on: the reading above is the server's time, not the code's.
        self._opened = time.monotonic()

    def _shut(self, now: float) -> None:
        pids = self._processes()
        if self._on_mark:
            self._on_mark("end", pids)
        self.time_s += now - self._opened
        self.count += 1
        peaks = {pid: st.get("VmHWM", 0) for pid in pids if (st := _status(pid))}
        if self._before_reset:
            for pid, hwm in peaks.items():
                self._before_reset(pid, hwm)
        if peaks:
            self.peak_kb = max(self.peak_kb, max(peaks.values()))
            total = sum(peaks.values())
            self.peak_total_kb = max(self.peak_total_kb, total)
            self.extra_kb = max(self.extra_kb, total - sum(self._base.get(p, 0) for p in peaks))

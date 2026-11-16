"""Mark the code a harness measures, so MIMIR times and weighs that alone:

    data = make_inputs()          # not measured
    with region():
        out = solve(data)         # measured: time_s, region_peak_rss_mb, region_extra_mb

Outside MIMIR (no ``MIMIR_REGION``) it does nothing. Copy it next to the harness.
"""
from __future__ import annotations

import contextlib
import fcntl
import os

_fds: tuple[int, int] | None = None
_off = False


def _channel() -> tuple[int, int] | None:
    global _fds, _off
    path = os.environ.get("MIMIR_REGION")
    if _fds or _off or not path:
        return _fds
    try:
        # Non-blocking open: with no reader (a rank on another node), give up at once.
        out = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        ack = os.open(path + ".ack", os.O_RDONLY | os.O_NONBLOCK)
        fcntl.fcntl(ack, fcntl.F_SETFL, fcntl.fcntl(ack, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        _fds = (out, ack)
    except OSError:
        _off = True
    return _fds


def mark(word: str) -> None:
    """Say ``begin`` or ``end``; return once MIMIR has read its clocks."""
    fds = _channel()
    if fds:
        try:
            os.write(fds[0], word.encode() + b"\n")
            os.read(fds[1], 1)
        except OSError:
            pass


@contextlib.contextmanager
def region():
    """The code in the ``with`` block is what MIMIR measures."""
    mark("begin")
    try:
        yield
    finally:
        mark("end")

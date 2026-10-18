"""Time spent waiting on a person, excluded from tool timeout budgets.

An approval card waits inside the tool call; counting it reported calls as timed out
before they ran. Thread-local, so sessions never see each other's pauses; reentrant.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Iterator

_local = threading.local()


@contextmanager
def human_pause() -> Iterator[None]:
    """Mark a block of code as waiting on the user, not on the tool."""
    depth = getattr(_local, "depth", 0)
    if depth == 0:
        _local.started = time.monotonic()
    _local.depth = depth + 1
    try:
        yield
    finally:
        _local.depth = depth
        if depth == 0:
            _local.total = getattr(_local, "total", 0.0) + (time.monotonic() - _local.started)
            _local.started = None


def elapsed() -> float:
    """Seconds this thread has spent waiting on the user, including a pause in progress.

    Monotonically increasing for the life of the thread; callers take differences
    across an interval rather than reading it as an absolute.
    """
    total = getattr(_local, "total", 0.0)
    if getattr(_local, "depth", 0) > 0:
        total += time.monotonic() - _local.started
    return total

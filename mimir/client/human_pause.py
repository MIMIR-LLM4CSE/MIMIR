"""Wall-clock time spent blocked on a human, excluded from tool timeout budgets.

A tool call's timeout budgets the *tool's own work*. An approval card, however, can
sit unanswered for as long as the user likes, and that wait happens inside the tool
call: the prompt is raised by the policy gate before the tool ever reaches its
server. Counting it against the budget reported commands as "timed out after 120s"
when they had not started running — the user simply took two minutes to click Allow.

Prompts block the agent's own thread (the WS worker thread, or the CLI main thread),
so the accounting is thread-local: concurrent sessions in one process never see each
other's pauses. Reentrant, so a prompt raised inside another prompt's wait is
counted once.
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

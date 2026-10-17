"""A worker's events on their way to the pump."""
from __future__ import annotations

import queue
from typing import Any, Callable


class Outbox(queue.Queue):
    """A worker's events on their way to the pump, which each one wakes.

    ``on_put`` is set by the pool once the worker is built, to a thread-safe nudge of
    the pump; until then, and once the loop is gone, the pump's own tick drains it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.on_put: Callable[[], None] | None = None

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        notify = self.on_put
        if notify is not None:
            try:
                notify()
            except RuntimeError:
                pass   # the loop is closing; there is no pump left to wake

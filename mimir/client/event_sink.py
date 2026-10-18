"""The sink the engine's UI events go to, bound per run in a ``ContextVar``.

Binding it once reaches every nested helper and ``asyncio.gather`` task; with none
bound, :func:`emit` prints a JSON line.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

# The active event sink for the current context. None -> print to stdout.
_current_sink: contextvars.ContextVar[Callable[[dict], None] | None] = (
    contextvars.ContextVar("agent_event_sink", default=None)
)


def emit(event: dict) -> None:
    """Deliver a structured event to the bound sink, or print it if none is bound.

    A sink that raises must never break the agent loop, so failures fall back to
    the default stdout path.
    """
    sink = _current_sink.get()
    if sink is None:
        print(json.dumps(event))
        return
    try:
        sink(event)
    except Exception:
        print(json.dumps(event))


def captured_emitter() -> Callable[[dict], None] | None:
    """The bound sink, captured here for a callback that fires in another task; None if unbound."""
    sink = _current_sink.get()
    if sink is None:
        return None

    def _emit(event: dict) -> None:
        try:
            sink(event)
        except Exception:
            logger.debug("_emit: a best-effort step failed and was skipped", exc_info=True)

    return _emit


def set_event_sink(sink: Callable[[dict], None]) -> Any:
    """Bind *sink* for the current context; returns a token for :func:`reset_event_sink`."""
    return _current_sink.set(sink)


def reset_event_sink(token: Any) -> None:
    """Restore the sink that was active before the matching :func:`set_event_sink`."""
    _current_sink.reset(token)


@contextlib.contextmanager
def event_sink(sink: Callable[[dict], None]) -> Iterator[None]:
    """Context manager that binds *sink* for the duration of the block."""
    token = set_event_sink(sink)
    try:
        yield
    finally:
        reset_event_sink(token)

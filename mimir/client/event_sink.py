"""Injectable event sink for the agent engine.

The engine emits structured UI events (status, thinking, tool_call, tool_result,
diff). Historically these were written as ``print(json.dumps(...))``
to stdout and captured by a front-end that monkeypatched ``sys.stdout``. That
coupled the engine to a stdout wire protocol and made it impossible to embed or
test headless without capturing stdout.

This module provides a thin seam instead: emit sites call :func:`emit`, which
routes the event to a sink bound for the current run. The sink is held in a
``contextvars.ContextVar`` so it is copied into ``asyncio.gather`` tasks — binding
it once at the top of a run reaches every nested helper and parallel tool task
without threading a parameter through each function. When no sink is bound (e.g.
the CLI front-end), :func:`emit` falls back to printing a JSON line, preserving
the original behaviour exactly.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
from typing import Any, Callable, Iterator

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

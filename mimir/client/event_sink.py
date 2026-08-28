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


def captured_emitter() -> Callable[[dict], None] | None:
    """The currently bound sink, captured for emitting from another task later.

    :func:`emit` reads the ContextVar at call time, which is right for the agent loop
    and wrong for a callback that fires elsewhere: a task started before the sink was
    bound — an MCP session's receive loop, say — carries a snapshot of the context from
    back then, so emitting there would find no sink and print instead. Capture the sink
    where it IS bound, call the result where the event happens.

    Returns None when nothing is bound, which is also the answer to "is there a
    frontend to feed at all" — with no sink, a stream of events is noise, not a view.
    """
    sink = _current_sink.get()
    if sink is None:
        return None

    def _emit(event: dict) -> None:
        try:
            sink(event)
        except Exception:
            pass

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

"""The tool call whose execution is running in this context. Set around each call's task
by the dispatch, so a prompt raised from inside the call can say which call it holds up."""
from __future__ import annotations

from contextvars import ContextVar

CURRENT_CALL_ID: ContextVar[str | None] = ContextVar("mimir_current_call_id", default=None)

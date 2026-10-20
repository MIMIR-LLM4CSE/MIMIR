"""One finished tool call, as every observer reads it."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ...context.execution_context import ExecutionContext
from ...core.ports import ObserverAgent


@dataclass(frozen=True)
class Call:
    agent: ObserverAgent
    tool_name: str
    execution_context: ExecutionContext
    arguments: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)
    # The payload's ``status``: "ok", "error", or None for a result that is not JSON.
    status: Any = None
    # The workspace path the call targets, normalised; "" when it names none.
    path: str = ""
    call_id: str = ""


Observer = Callable[[Call], None]

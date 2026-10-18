"""An execution context built from flat field names, for tests that set a few fields."""
from __future__ import annotations

from typing import Any

from mimir.client.context.execution_context import ExecutionContext


def build_execution_context(**values: Any) -> ExecutionContext:
    """A fresh context with each named field set, wherever its record keeps it."""
    context = ExecutionContext()
    for name, value in values.items():
        setattr(context.home(name), name, value)
    return context

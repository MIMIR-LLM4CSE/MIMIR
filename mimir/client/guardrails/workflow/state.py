"""The workflow state (discover, edit, validate, conclude) and what still owes a check."""
from __future__ import annotations

from ...context.context_view import ContextView
from ...context.execution_context import ExecutionContext
import warnings


WORKFLOW_STATES: tuple[str, ...] = ("discover", "edit", "validate", "conclude")


def pending_validation_paths(execution_context: ContextView) -> list[str]:
    """Dirty files that still owe a check, except those that cannot be read as text."""
    dirty_files = set(execution_context.edits.dirty_written_files)
    validated_files = set(execution_context.checks.validated_files)
    unverifiable = set(execution_context.checks.unverifiable_files or set())
    return sorted(
        path for path in dirty_files
        if path not in validated_files and path not in unverifiable
    )


def has_pending_validation(execution_context: ContextView) -> bool:
    return bool(pending_validation_paths(execution_context))


def has_blocking_denials(execution_context: ContextView) -> bool:
    return bool(execution_context.denials.denied_tool_calls)


def set_workflow_state(execution_context: ExecutionContext, new_state: str) -> None:
    if new_state in WORKFLOW_STATES:
        execution_context.workflow_state = new_state
    else:
        warnings.warn(
            f"set_workflow_state called with unknown state '{new_state}'; ignored. "
            f"Expected one of: {', '.join(WORKFLOW_STATES)}",
            stacklevel=2,
        )

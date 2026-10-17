"""The workflow state (discover, edit, validate, conclude) and what still owes a check."""
from __future__ import annotations

import warnings
from ...context.execution_context import ExecutionContext


WORKFLOW_STATES: tuple[str, ...] = ("discover", "edit", "validate", "conclude")


def pending_validation_paths(execution_context: ExecutionContext) -> list[str]:
    """Dirty files that still owe a check, minus the ones nothing here can read.

    The mandatory axis is the cheap one — does this parse, is it whole — and since it is
    performed in-process (``guardrails.builtin_check``) it is possible everywhere, for
    every language. What leaves this list for the ledger is now only a file the floor
    cannot read as text at all: a binary, or bytes that are not UTF-8.
    """
    dirty_files = set(execution_context.dirty_written_files)
    validated_files = set(execution_context.validated_files)
    unverifiable = set(execution_context.unverifiable_files or set())
    return sorted(
        path for path in dirty_files
        if path not in validated_files and path not in unverifiable
    )


def has_pending_validation(execution_context: ExecutionContext) -> bool:
    return bool(pending_validation_paths(execution_context))


def has_blocking_denials(execution_context: ExecutionContext) -> bool:
    return bool(execution_context.denied_tool_calls)


def set_workflow_state(execution_context: ExecutionContext, new_state: str) -> None:
    if new_state in WORKFLOW_STATES:
        execution_context.workflow_state = new_state
    else:
        warnings.warn(
            f"set_workflow_state called with unknown state '{new_state}'; ignored. "
            f"Expected one of: {', '.join(WORKFLOW_STATES)}",
            stacklevel=2,
        )

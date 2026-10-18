"""What the model said it would do this turn, and the checklist steps still open."""
from __future__ import annotations

from ...context.todo import load_todo_items
from ...context.execution_context import (
    unwritten_declared_files,
)
from ...context.execution_context import ExecutionContext


def turn_made_commitments(execution_context: ExecutionContext) -> bool:
    """Did this turn commit to producing something: an edit, a declared target, a checklist?

    All three are per-query fields, so no earlier turn's intent carries over.
    """
    return bool(
        execution_context.edits.code_mutation_started
        or execution_context.edits.declared_edit_set
        or execution_context.plan.todo_written
    )


def unhonoured_commitments(execution_context: ExecutionContext) -> bool:
    """True when this turn promised work and some of it is still undone.

    The two halves of "promised and then skipped" that the codebase already defines
    separately — files named and never written, and required checklist steps left
    unticked — behind the one question every caller was reconstructing by hand.
    """
    if not turn_made_commitments(execution_context):
        return False
    if unwritten_declared_files(execution_context):
        return True
    return any(
        not it.get("optional") for it in unchecked_checklist_items(execution_context)
    )


def unchecked_checklist_items(execution_context: ExecutionContext) -> list[dict]:
    """Unticked steps of the live checklist, optional ones included; [] without a checklist."""
    todo_fp = execution_context.plan.todo_file_path
    if not todo_fp:
        return []
    # Imported lazily: the prompt package pulls in the extension loader, and the
    # guardrails layer is imported during its construction.
    return [it for it in load_todo_items(todo_fp) if not it.get("done")]

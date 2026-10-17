"""What the model said it would do this turn, and the checklist steps still open."""
from __future__ import annotations

from ...context.todo import load_todo_items
from ...context.execution_context import (
    unwritten_declared_files,
)
from ...context.execution_context import ExecutionContext


def turn_made_commitments(execution_context: ExecutionContext) -> bool:
    """Did THIS turn commit to producing something?

    Not ``code_mutation_started``, which says a code file was edited: a turn that writes
    a checklist, declares eight files and ends without writing any of them committed to
    work it did not do, and the completeness guards must see that.

    Three signals, all per-query by declaration (``_NO_TRAITS`` in ``_FIELD_SPECS``), so
    none of them can carry a previous turn's intent into this one: an edit happened, a
    set of target files was declared, or a checklist was written. A genuine
    discovery-only turn exhibits none and is still never nudged — that exemption now
    falls out of the definition instead of being approximated.
    """
    return bool(
        execution_context.code_mutation_started
        or execution_context.declared_edit_set
        or execution_context.todo_written
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
    """Steps still unticked on the live checklist, or [] if there is no checklist.

    The single reader of checklist state outside the prompt builder, shared by the
    completion issues below, the finalization predicate, and the unfinished-plan
    nudge. Fails closed to ``[]`` on a missing or unreadable file so a run without a
    checklist — the majority — keeps behaving exactly as before, rather than having
    an obligation invented for it.

    Optional steps are included; callers that must not block on them filter on
    ``item["optional"]``.
    """
    todo_fp = execution_context.todo_file_path
    if not todo_fp:
        return []
    # Imported lazily: the prompt package pulls in the extension loader, and the
    # guardrails layer is imported during its construction.
    return [it for it in load_todo_items(todo_fp) if not it.get("done")]

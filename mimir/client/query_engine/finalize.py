"""End-of-query bookkeeping: annotate the answer, persist memory, finalize.

``_finalize_answer`` is the single exit funnel for both loops (annotate written
files → persist a compact memory note → return). Extracted from ``agent_loop.py``.
"""
from __future__ import annotations

from typing import Any

from ..context import validation_tier
from ..guardrails.workflow import unchecked_checklist_items
from ..prompt.system_prompt import auto_store_memory


#: How a file reached the ``oracle`` tier. Derived, never stored: a file promoted by
#: red→green is exactly one an ``executed``-tier check was seen failing on, and that
#: set is already kept for the promotion itself. Anything else at ``oracle`` got there
#: by reporting a numerical invariant.
_ORACLE_DISCRIMINATED = "red→green"
_ORACLE_INVARIANT = "reported invariant"


def _oracle_basis(execution_context: dict, path: str) -> str:
    if path in (execution_context.get("executed_failures") or set()):
        return _ORACLE_DISCRIMINATED
    return _ORACLE_INVARIANT


def _annotate_answer_with_changes(answer: str, execution_context: dict) -> str:
    """Append a machine-recorded verification ledger to the answer text.

    Saved into conversation history so the next query — and the user — sees what
    was actually established, independent of whatever prose the model chose for
    its closing answer. That independence is the entire point: the model's summary
    and the recorded evidence used to sit side by side with nothing reconciling
    them, so a run could assert "verified" directly above a list of files that had
    only ever been executed, never checked against anything.

    Report-only, and emitted after the model has stopped acting: it cannot loop,
    cannot be argued with, and blocks nothing. Lines with no content are omitted,
    so an all-green run with no checklist collapses to the original one-line form.
    """
    written = sorted(execution_context.get("dirty_written_files", set()))
    validated = execution_context.get("validated_files", set())
    unchecked = unchecked_checklist_items(execution_context)
    declared = execution_context.get("declared_edit_set", set()) or set()
    unwritten = sorted(declared - set(written))

    if not written and not unchecked and not unwritten:
        return answer

    lines: list[str] = []
    if written:
        parts = []
        for f in written:
            if f not in validated:
                parts.append(f"{f} (NOT validated)")
            else:
                tier = validation_tier(execution_context, f) or "executed"
                if tier == "oracle":
                    tier = f"oracle — {_oracle_basis(execution_context, f)}"
                parts.append(f"{f} (validated: {tier})")
        lines.append("Files written: " + " ; ".join(parts))
        # The `oracle` tier means a check compared the artifact against something the
        # code does not itself define. Saying which way it got there, plainly, is what
        # stops "the tests passed" from being read back as "the result is correct".
        #
        # The absence line is stated in *domain-neutral* terms on purpose. It used to
        # name reference comparisons, conservation checks and convergence measurements
        # — vocabulary a parser, a CLI or a string refactor can never satisfy, so it
        # fired on every non-numerical run and became wallpaper. That is how a ledger
        # dies. Numerical wording now appears only to *qualify* an invariant that was
        # actually reported, never to announce its absence.
        #
        # Only worth saying once code was actually *run*: if nothing got past
        # syntax/static checks the per-file tier already tells the whole story.
        # (Docs/config never reach here — non-code paths are excluded upstream.)
        tiers = [validation_tier(execution_context, f) for f in written]
        if "oracle" not in tiers and "executed" in tiers:
            lines.append(
                "Discrimination: none observed — every check that passed was first run "
                "after the change and was never seen failing, so nothing establishes "
                "that it tells working code from broken code."
            )
        elif any(_oracle_basis(execution_context, f) == _ORACLE_INVARIANT
                 for f in written if validation_tier(execution_context, f) == "oracle"):
            lines.append(
                "Reported invariant: a validation run printed a numerical invariant. Its "
                "presence is recorded, never its value — nothing here compared it against "
                "a sealed reference."
            )
    if unwritten:
        lines.append("Declared but never written: " + ", ".join(unwritten))
    if unchecked:
        required = [it for it in unchecked if not it.get("optional")]
        optional = [it for it in unchecked if it.get("optional")]
        if required:
            preview = "; ".join(it["text"] for it in required[:2])
            more = f" (+{len(required) - 2} more)" if len(required) > 2 else ""
            lines.append(f"Checklist: {len(required)} step(s) unchecked — {preview}{more}")
        if optional:
            lines.append(f"Checklist: {len(optional)} optional step(s) not done")

    if not lines:
        return answer
    return (
        answer
        + "\n\n[Verification ledger — machine-recorded, not model-authored]\n"
        + "\n".join(lines)
    )


async def _persist_answer(
    agent: Any,
    query: str,
    answer: str,
    execution_context: dict,
    logger: Any,
) -> None:
    """Persist a compact memory note for this exchange."""
    await auto_store_memory(
        query=query,
        answer=answer,
        tool_owner=agent.tool_owner,
        run_tool=lambda tool, args, context: agent._run_tool(tool, args, execution_context=context),
        truncate_text=agent._truncate_text,
        execution_context=execution_context,
        logger=logger,
    )


async def _finalize_answer(
    agent: Any,
    query: str,
    answer: str,
    execution_context: dict,
    messages: list[dict],
    logger: Any,
) -> str:
    """Common end-of-query bookkeeping shared by every exit path.

    Annotates the answer with the file-change record, persists a compact memory note,
    saves the carry context for the next query, and stashes the full message list (minus
    the system prompt) for full-context history. Returns the annotated answer.
    """
    answer = _annotate_answer_with_changes(answer, execution_context)
    await _persist_answer(agent, query, answer, execution_context, logger)
    agent._update_carry_context(execution_context)
    agent._last_full_messages = messages[1:]
    return answer

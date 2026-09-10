"""End-of-query bookkeeping: annotate the answer, persist memory, finalize.

``_finalize_answer`` is the single exit funnel for both loops (annotate written
files → persist a compact memory note → return). Extracted from ``agent_loop.py``.
"""
from __future__ import annotations

from typing import Any

from ..prompt.system_prompt import auto_store_memory
from .verification import build_ledger, render_ledger


def _annotate_answer_with_changes(answer: str, execution_context: dict) -> str:
    """Append the machine-recorded verification ledger to the answer text.

    Saved into conversation history so the next query — and the user — sees what was
    actually established, independent of whatever prose the model chose to close with.
    That independence is the point: a run could otherwise assert "verified" directly
    above files that had only ever been executed, never checked against anything.

    See :mod:`.verification` for the ledger's shape and its marker contract with the
    front-ends. A run with nothing to record leaves the answer untouched.
    """
    ledger = build_ledger(execution_context)
    if ledger is None:
        return answer
    return answer + render_ledger(ledger)


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
    agent._last_turn_start = _turn_start_index(agent, messages)
    return answer


def _turn_start_index(agent: Any, messages: list[dict]) -> int | None:
    """Where this turn's own messages begin in ``_last_full_messages``, or None.

    Resolved by identity against the message the loop recorded as this turn's opening
    (see ``_run_agent_loop``), because that is the only thing about the list that the
    in-turn budget rewrites cannot invalidate: they evict, summarize and truncate
    messages, but the object a surviving entry *is* does not change.

    Returned as an index into ``messages[1:]`` — the system message is not exposed —
    and pointing one past the opening message, since a caller that appended the query
    to its own record before submitting has already stored everything up to it.

    ``None`` says the opening message is gone: a long turn can have its own start
    summarized away by the compaction pass, and there is then no honest boundary to
    give. A caller must fall back rather than guess, which is the whole point of
    saying so instead of returning a number that looks usable.
    """
    opening = getattr(agent, "_turn_opening_message", None)
    if opening is None:
        return None
    for i, m in enumerate(messages):
        if m is opening:
            return i  # index in messages[1:] of the message *after* the opening one
    return None

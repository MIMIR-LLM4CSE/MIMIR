"""End-of-query bookkeeping: annotate the answer, finalize.

``_finalize_answer`` is the single exit funnel for both loops (annotate written
files → save carry context → return). Extracted from ``agent_loop.py``.

Nothing is written to persistent memory here: what is worth remembering is the
model's call, made during the run (see ``_SECTION_MEMORY``). A note stored on its
behalf at every exit doubled what it had already written.
"""
from __future__ import annotations

from typing import Any

from .history import reconcile_tool_pairs
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


async def _finalize_answer(
    agent: Any,
    query: str,
    answer: str,
    execution_context: dict,
    messages: list[dict],
    logger: Any,
) -> str:
    """Common end-of-query bookkeeping shared by every exit path.

    Annotates the answer with the file-change record, saves the carry context for the
    next query, and stashes the full message list (minus the system prompt) for
    full-context history. Returns the annotated answer. ``query`` and ``logger`` are
    kept so every exit path calls it the same way.
    """
    answer = _annotate_answer_with_changes(answer, execution_context)
    agent._update_carry_context(execution_context)
    _record_turn(agent, messages)
    return answer


def _record_turn(agent: Any, messages: list[dict], *, repair: bool = True) -> None:
    """Stash this turn's transcript (minus the system prompt) for full-context history.

    Called on every exit, not only on a finished answer. The front-end reads it back
    once the turn is over, whatever the outcome, and replaces its history with it. A
    turn that was cancelled or raised used to leave the *previous* turn's transcript
    here, so the history rolled back to it: the cancelled turn's own work was lost, its
    messages were archived a second time, and a "continue" replayed the same steps
    word for word. After a session switch the stale transcript was another session's.

    *repair* is off when a turn claims the slot on entry: a resumed turn has just taken
    out the results it is about to produce, and a repair would stub them in first.
    """
    if repair:
        messages[:] = reconcile_tool_pairs(messages)
    agent._last_turn_start = _turn_start_index(agent, messages)
    _drop_earlier_reasoning(messages, agent._last_turn_start)
    agent._last_full_messages = messages[1:]


def _drop_earlier_reasoning(messages: list[dict], turn_start: int | None) -> None:
    """Strip the reasoning of every step before this turn's opening message.

    Reasoning matters inside the turn that produced it: it is what the next step builds
    on. Across turns it only grows the prompt, which is why most chat templates drop it
    there anyway. Replaced by copies, not popped, so the caller's dicts are untouched.
    """
    if turn_start is None:
        return
    for i in range(1, turn_start + 1):
        m = messages[i]
        if "reasoning" in m:
            messages[i] = {k: v for k, v in m.items() if k != "reasoning"}


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

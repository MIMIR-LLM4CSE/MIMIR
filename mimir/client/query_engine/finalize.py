"""End-of-query bookkeeping: ``finalize_answer`` is the exit funnel of both loops.

Nothing goes to persistent memory here: that is the model's call (``_SECTION_MEMORY``).
"""
from __future__ import annotations

from typing import Any

from ..guardrails.nudges import exercise_route
from ..guardrails.verification.ladder import record_exercise_obstacle
from ..protocol.messages import reconcile_tool_pairs
from ..guardrails.verification.ledger import build_ledger, render_ledger
from ..context.execution_context import ExecutionContext
from ..core.ports import AgentPort


def _annotate_answer_with_changes(answer: str, execution_context: ExecutionContext) -> str:
    """Append the verification ledger to the answer (see :mod:`.verification`), if there is one."""
    ledger = build_ledger(execution_context)
    if ledger is None:
        return answer
    return answer + render_ledger(ledger)


async def finalize_answer(
    agent: AgentPort,
    query: str,
    answer: str,
    execution_context: ExecutionContext,
    messages: list[dict],
    logger: Any,
) -> str:
    """Annotate the answer, save the carry context, record the turn; return the answer."""
    # Asked where the ledger is built, for the case it reports: code written, nothing run.
    record_exercise_obstacle(execution_context, lambda: exercise_route(agent, execution_context)[1])
    answer = _annotate_answer_with_changes(answer, execution_context)
    agent.carry.update(execution_context)
    record_turn(agent, messages)
    return answer


def record_turn(agent: AgentPort, messages: list[dict], *, repair: bool = True) -> None:
    """Stash this turn's transcript (minus the system prompt) for full-context history.

    On every exit, so a cancelled or failed turn keeps its own work. *repair* is off for a
    resumed turn that has just taken out the results it will produce.
    """
    if repair:
        messages[:] = reconcile_tool_pairs(messages)
    agent.turn.last_turn_start = _turn_start_index(agent, messages)
    _drop_earlier_reasoning(messages, agent.turn.last_turn_start)
    agent.turn.last_full_messages = messages[1:]


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


def _turn_start_index(agent: AgentPort, messages: list[dict]) -> int | None:
    """Where this turn's messages begin in ``messages[1:]`` (one past the opening), or None.

    Found by identity of the opening message, which survives eviction and compaction;
    None when the compaction summarized it away, and the caller must fall back.
    """
    opening = agent.turn.opening_message
    if opening is None:
        return None
    for i, m in enumerate(messages):
        if m is opening:
            return i  # index in messages[1:] of the message *after* the opening one
    return None

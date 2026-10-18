"""One query, from the user's message to the answer.

:func:`run_agent_query` builds the turn — execution context, system prompt, the
message list, a ``/skill`` the user invoked — and :func:`run_loops` runs it through
the loop of its mode, then through whichever loop the mode is switched to.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from ..context import validate_execution_context
from ..context.execution_context import ExecutionContext
from ..core.ports import AgentPort
from ..core.settings import normalize_mode
from ..event_sink import emit
from ..protocol.messages import reconcile_tool_pairs
from .agent_loop import AgentLoop
from .deferral import KIND_CALLS, KIND_PLAN_DECISION, take_deferred_calls
from .finalize import record_turn
from .loop_base import ModeSwitch, StepLoop, TurnContext
from .plan_loop import PlanLoop
from .verification import build_ledger

logger = logging.getLogger(__name__)


def _loop_for(turn: TurnContext, mode: str, system_content: str, *,
              resume_decision: bool = False) -> StepLoop:
    if mode == "plan":
        return PlanLoop(turn, mode, system_content, resume_decision=resume_decision)
    return AgentLoop(turn, mode, system_content)


async def run_loops(turn: TurnContext, mode: str, system_content: str, *,
                    resume_calls: list | None = None,
                    resume_decision: bool = False) -> str:
    """Run *turn* in the loop of *mode*, following every mode switch to its answer.

    A resumed turn (see ``deferral``) first runs the calls the user has now answered,
    as the step they were set aside from, or reopens the plan decision it left open.
    """
    loop = _loop_for(turn, mode, system_content, resume_decision=resume_decision)
    if resume_calls:
        ended = await loop.dispatch(resume_calls)
        if ended is not None:
            return ended
    while True:
        outcome = await loop.run()
        if not isinstance(outcome, ModeSwitch):
            return outcome
        loop = _loop_for(turn, outcome.mode, outcome.system_content)


def _emit_interrupted_state(execution_context: ExecutionContext) -> None:
    """Say what an abandoned query left behind, best-effort and never raising.

    Best-effort because it runs while an exception is in flight: a failure here
    would replace the real error with a worse one.
    """
    try:
        ledger = build_ledger(execution_context)
        if ledger is None:
            return
        emit({
            "type": "status",
            "text": "  ⚠ Query ended early — state left behind: " + ledger["summary"],
        })
        for row in ledger["rows"]:
            emit({"type": "status", "text": f"    · {row}"})
    except Exception:  # noqa: BLE001 — never mask the failure being reported
        logger.debug("_emit_interrupted_state: a best-effort step failed and was skipped", exc_info=True)




async def run_agent_query(
    *,
    agent: AgentPort,
    query: str,
    max_steps: int = 0,
    history: list[dict] | None = None,
    mode: str | None = None,
    thinking: bool = False,
    streaming: bool = True,
    logger: Any = None,
    token_callback: Any = None,
    think_token_callback: Any = None,
    think_start_callback: Any = None,
    think_end_callback: Any = None,
    resume: dict | None = None,
) -> str:
    """Run one user query through the mode loops; return the final answer.

    *resume* picks up a deferred turn (see ``deferral``): no new user message, the deferred
    calls first.
    """
    agent.turn.tool_cache = {}
    execution_context = agent.new_execution_context()
    # Set here rather than passed in: a front-end's stand-in agent supplies its own
    # zero-argument factory, and the root is a fact about the agent either way.
    execution_context.bookkeeping.workspace_root = getattr(agent, "workspace_root", "")
    agent.carry.apply(execution_context)
    validate_execution_context(execution_context)
    active_mode = normalize_mode(mode or agent.mode)
    # Baseline for the live-mode check: only a *later* change to agent.mode is a
    # user switching mid-run. Seeded from the agent, not from active_mode, so an
    # explicit per-query override never registers as one. See _live_mode.
    execution_context.bookkeeping.observed_agent_mode = agent.mode

    system_content = await agent.build_system_content(active_mode=active_mode)

    # Store the active todo file path so _sync_checklist can watch it for changes.
    execution_context.plan.todo_file_path = agent.todo_file()

    # Append the query unless history already ends with it (the WS server appends it, the CLI
    # does not): two identical user turns break strict tokenizers.
    hist = list(history or [])
    messages: list[dict] = [
        {
            "role": "system",
            "content": system_content,
        },
        *hist,
    ]
    if resume is None and not (
        hist
        and hist[-1].get("role") == "user"
        and hist[-1].get("content") == query
    ):
        messages.append({"role": "user", "content": query})

    # Inherited history can arrive with a broken assistant↔tool pairing: a front-end
    # trims by token budget and can pop an assistant turn while keeping its results.
    # Normalise once here so plan mode — which does no budget enforcement of its own —
    # is covered too, and no backend ever has to cope with a dangling pair.
    messages[:] = reconcile_tool_pairs(messages)

    # The opening message of this turn, handed over as the object, never as an index: the
    # budget rewrites this list in place, and ``finalize_answer`` resolves it at the end.
    # Taken before the placeholders of a resume are fixed.
    resume_calls = (
        take_deferred_calls(messages, list(resume.get("call_ids") or []))
        if resume is not None and resume.get("kind") == KIND_CALLS else []
    )
    agent.turn.opening_message = messages[-1] if messages else None
    # This turn owns the transcript slot from here on: an earlier turn's (or an earlier
    # session's) must never be read back as this one's, whatever way this one ends.
    record_turn(agent, messages, repair=False)

    # Every loop below mutates this exact list in place, so handing the reference out
    # is enough for a front-end to read the in-flight context without polling the agent.
    agent.turn.live_messages = messages

    # -------------------------------------------------------
    # Skill invoked by the user: /skill-name
    # -------------------------------------------------------
    # The user decided before the first step, so it is folded into messages[0]; a skill the
    # model needs is pulled mid-run with ``load_skill`` instead.
    skill_name = None
    if query.strip().startswith("/"):
        candidate = query.strip()[1:].split()[0]
        if getattr(agent, "skills", None) and candidate in agent.skills:
            skill_name = candidate

    if skill_name:
        skill = agent.skills[skill_name]
        _base_system_content = system_content
        # Folded into messages[0], never a second ``system`` message (which accumulated in history
        # and meant different things per backend).
        system_content = (
            system_content
            + "\n\nSKILL CONTEXT (SUBORDINATE). "
            "The base system instructions remain fully authoritative. "
            "Apply this methodology where relevant. "
            + skill["content"]
        )
        messages[0]["content"] = system_content
        execution_context.bookkeeping.skill_suffix = system_content[len(_base_system_content):]
        # Counted among the loaded skills, with no call id: there is no tool result to
        # pin because the body is in messages[0], which nothing evicts. What this buys
        # is that the model cannot pull what it has already been given, and that the
        # cap counts a body the user put there the same as one the model asked for.
        execution_context.bookkeeping.skills_loaded[skill_name] = ""

    # messages[0] now carries the checklist as it stands at query start; _sync_checklist
    # rewrites it from here on, and only when the file behind it actually changes. The
    # mtime is seeded rather than left unset so an unchanged checklist costs nothing on
    # the first step of every query.
    _todo_fp = execution_context.plan.todo_file_path
    if _todo_fp:
        try:
            execution_context.plan.checklist_stamp = os.stat(_todo_fp).st_mtime_ns
        except OSError:
            pass

    turn = TurnContext(
        agent=agent, query=query, messages=messages,
        execution_context=execution_context, max_steps=max_steps,
        thinking=thinking, streaming=streaming, logger=logger,
        callbacks={
            "token_callback": token_callback,
            "think_token_callback": think_token_callback,
            "think_start_callback": think_start_callback,
            "think_end_callback": think_end_callback,
        },
    )

    try:
        return await run_loops(
            turn, active_mode, system_content, resume_calls=resume_calls,
            resume_decision=(resume is not None and resume.get("kind") == KIND_PLAN_DECISION),
        )
    except Exception:
        # A query that dies mid-flight still emits the ledger of where it left the workspace;
        # the exception keeps propagating.
        _emit_interrupted_state(execution_context)
        raise
    finally:
        agent.turn.live_messages = None
        # A cancelled or failed turn never reaches finalize_answer; record what it did
        # so the front-end keeps its work instead of rolling back to the turn before.
        record_turn(agent, messages)

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
from .finalize import _record_turn
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
    """Run one user query through plan/agent modes with policy guardrails.

    *resume* picks up a turn that was set aside while waiting on the user (see
    ``deferral``): no new user message, the deferred calls run first with the answer
    the front-end already holds, and the loop carries on from there.

    Builds the per-query execution context and system prompt, then dispatches to the
    plan-mode or agent-mode loop. The two loops (and every exit path) share
    ``_finalize_answer`` for end-of-query bookkeeping.
    """
    agent.turn.tool_cache = {}
    execution_context = agent.new_execution_context()
    # Set here rather than passed in: a front-end's stand-in agent supplies its own
    # zero-argument factory, and the root is a fact about the agent either way.
    execution_context.workspace_root = getattr(agent, "workspace_root", "")
    agent.apply_carry_context(execution_context)
    validate_execution_context(execution_context)
    active_mode = normalize_mode(mode or agent.mode)
    # Baseline for the live-mode check: only a *later* change to agent.mode is a
    # user switching mid-run. Seeded from the agent, not from active_mode, so an
    # explicit per-query override never registers as one. See _live_mode.
    execution_context.observed_agent_mode = getattr(agent, "mode", active_mode)

    system_content = await agent.build_system_content(active_mode=active_mode)

    # Store the active todo file path so _sync_checklist can watch it for changes.
    execution_context.todo_file_path = agent.todo_file()

    # The caller may or may not have already appended the current user turn to
    # ``history``: the WS server does (it owns the chat history and appends the
    # message before dispatching), the CLI does not. Only add the query ourselves
    # when history doesn't already end with this exact user message — otherwise the
    # prompt carries two consecutive identical ``user`` turns. Tolerant templates
    # ignore that, but a strict tokenizer treats the illegal role sequence as garbage
    # and degenerates into token salad.
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

    # The message this turn opens on — the caller's own last archived message, or the
    # query just appended above. Handed over as the *object*, never as an index:
    # ``_enforce_context_budget`` rewrites this list in place while the turn runs
    # (evicting old tool results, replacing the middle with a summary, then repairing
    # the assistant↔tool pairing those break), so a position computed before the turn
    # stops meaning what it meant. ``_finalize_answer`` resolves it back to an index at
    # the one moment both are true together. A caller that slices on its own arithmetic
    # instead re-archives or drops whatever the rewrite shifted — and, because the
    # repair leaves no orphan behind, cuts straight through an assistant↔tool pair.
    # Taken before the opening is fixed: the placeholders are not part of the record.
    resume_calls = (
        take_deferred_calls(messages, list(resume.get("call_ids") or []))
        if resume is not None and resume.get("kind") == KIND_CALLS else []
    )
    agent.turn.opening_message = messages[-1] if messages else None
    # This turn owns the transcript slot from here on: an earlier turn's (or an earlier
    # session's) must never be read back as this one's, whatever way this one ends.
    _record_turn(agent, messages, repair=False)

    # Every loop below mutates this exact list in place, so handing the reference out
    # is enough for a front-end to read the in-flight context without polling the agent.
    agent.turn.live_messages = messages

    # -------------------------------------------------------
    # Skill invoked by the user: /skill-name
    # -------------------------------------------------------
    # The only skill this function resolves. A skill the MODEL decides it needs is
    # loaded mid-run through the ``load_skill`` tool, whose result stays in the tail
    # and is pinned there — the index of what is loadable is in the system prompt, so
    # the choice is made by whoever has read the code, at the step where that becomes
    # clear. An explicit slash command is different in kind: the user has already
    # decided, before the first step, so it is folded into messages[0] where it costs
    # the prefix nothing and binds every step of the query.
    skill_name = None
    if query.strip().startswith("/"):
        candidate = query.strip()[1:].split()[0]
        if getattr(agent, "skills", None) and candidate in agent.skills:
            skill_name = candidate

    if skill_name:
        skill = agent.skills[skill_name]
        _base_system_content = system_content
        # Folded into messages[0] rather than appended as a second `system` message.
        # Appending made it accumulate: it is written back into session history, so the
        # next query appended another copy — one session reached nine, ~4.3k tokens of
        # the same text. It also meant the same history had two different meanings
        # depending on the backend, since the Anthropic path hoists every `system`
        # message into the top-level prompt while the vLLM path leaves it inline.
        # messages[0] is already rewritten in place on a mode switch and on thinking
        # sync, so the prefix-cache break this costs is one the query already takes.
        system_content = (
            system_content
            + "\n\nSKILL CONTEXT (SUBORDINATE). "
            "The base system instructions remain fully authoritative. "
            "Apply this methodology where relevant. "
            + skill["content"]
        )
        messages[0]["content"] = system_content
        execution_context.skill_suffix = system_content[len(_base_system_content):]
        # Counted among the loaded skills, with no call id: there is no tool result to
        # pin because the body is in messages[0], which nothing evicts. What this buys
        # is that the model cannot pull what it has already been given, and that the
        # cap counts a body the user put there the same as one the model asked for.
        execution_context.skills_loaded[skill_name] = ""

    # messages[0] now carries the checklist as it stands at query start; _sync_checklist
    # rewrites it from here on, and only when the file behind it actually changes. The
    # mtime is seeded rather than left unset so an unchanged checklist costs nothing on
    # the first step of every query.
    _todo_fp = execution_context.todo_file_path
    if _todo_fp:
        try:
            execution_context.checklist_stamp = os.stat(_todo_fp).st_mtime_ns
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
        # A query that dies mid-flight leaves the workspace wherever it stood, and
        # the one thing nobody could reconstruct afterwards is *where* that was.
        # Observed at the end of session ``7d322a3b``: the provider error was
        # rendered as the whole answer, while a source file edited ninety seconds
        # earlier had never been measured and an optimisation session sat open on
        # a baseline nobody would remember. Same ledger the normal exit funnel
        # renders — this path simply never reached it.
        #
        # Emitted, never swallowed: the exception is the outcome and it keeps
        # propagating. This only means the outcome arrives with its state.
        _emit_interrupted_state(execution_context)
        raise
    finally:
        agent.turn.live_messages = None
        # A cancelled or failed turn never reaches _finalize_answer; record what it did
        # so the front-end keeps its work instead of rolling back to the turn before.
        _record_turn(agent, messages)

"""Plan mode: gather read-only evidence, record a plan, get approval, execute.

``_run_plan_mode`` mirrors the agent loop but with plan tools + the PLAN_BLOCKED/
PLAN_READONLY guards and an evidence gate; on approval it tail-calls
``_run_agent_loop`` (lazy import to avoid the agent_loop↔plan_loop cycle). Extracted
from ``agent_loop.py``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..event_sink import emit
from ..context.execution_context import has_discovery_evidence
from ..context.signals import query_requires_repo_discovery
from ..config.models import resolve_enforcement
from ..config.constants import (
    DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN, THINKING_DEPTH_AUTO, max_tools_for,
)
from ..guardrails.workflow import (
    PLAN_TODO_NUDGE_EARLY,
    PLAN_TODO_NUDGE_LATE,
    PLAN_DELIVER_ANSWER,
    PLAN_DELIVER_ANSWER_FIRM,
    PLAN_ALREADY_RECORDED_ERROR,
    PLAN_EVIDENCE_NUDGE,
    PLAN_APPROVED_EXECUTE,
    PLAN_REWORK_NUDGE,
    PLAN_REJECTED_STOP,
    PLAN_REJECTED_ANSWER,
    plan_revision_nudge,
)
from .streaming import _stream_chat, _process_response, _to_dict
from .dispatch import _dispatch_tool_calls
from .finalize import _finalize_answer
from .readonly_guard import filter_readonly_tool_calls
from .toollist import tools_for_plan_mode, cap_tools_by_relevance


_PLAN_ACCEPT = "Accept & start"
_PLAN_REJECT = "Reject"
_PLAN_REWORK = "Rework"

# Once the plan is recorded the model owes a prose answer + the approval prompt, but
# some models keep re-reading and re-echoing the plan instead — and the deliver nudge
# never breaks that, since a turn that calls tools never reaches the delivery branch.
# After this many further tool-calling turns its calls are dropped.
_PLAN_POST_RECORD_TOOL_TURNS = 2


def _reject_stalled_calls(tool_calls: list[Any], messages: list[dict]) -> None:
    """Drop *tool_calls* made after the plan was recorded, telling the model why.

    Mirrors ``filter_readonly_tool_calls``' convention: one ``role="tool"`` reply per
    dropped call so the transcript stays well-formed and the model sees the reason.
    """
    for tc in tool_calls:
        name = _to_dict(_to_dict(tc).get("function", {})).get("name", "")
        emit({"type": "status", "text": f"  ⏭ Plan already recorded — skipped '{name}'"})
        messages.append({"role": "tool", "content": json.dumps(
            {"status": "error", "error": PLAN_ALREADY_RECORDED_ERROR})})


def _clear_recorded_plan(execution_context: dict) -> None:
    """Forget the recorded plan so a reworked one has to be written afresh.

    The loop reads "is a plan recorded" from the execution context, so resetting only
    its local flag would leave the sticky context flags set and the very next tool turn
    would count the discarded plan as the new one.
    """
    execution_context["todo_written"] = False
    execution_context["plan_written"] = False


async def _request_plan_decision(agent: Any) -> tuple[str, str]:
    """Ask the user to accept, reject, or rework the proposed plan.

    Reuses the interactive ``_request_user_question`` prompt (wired to the CLI and
    the VS Code webview), which always offers a free-text "Other" entry on top of the
    listed choices. Returns ``(decision, feedback)`` where ``decision`` is one of
    ``"accept"``, ``"reject"``, ``"rework"``, ``"revise"`` (free-text = fold these
    changes in), or ``"none"``. ``"none"`` is returned when there is no interactive
    front-end (default shim / sub-agents / tests) or the user dismisses the prompt,
    so plan mode falls back to simply delivering the plan.
    """
    ask = getattr(agent, "_request_user_question", None)
    if not callable(ask):
        return "none", ""
    try:
        # The shim blocks (queue wait / CLI input); keep the event loop responsive.
        resp = await asyncio.to_thread(
            ask,
            [{
                "question": (
                    "Review the proposed plan. Accept it to execute in agent mode, reject it to stop "
                    "here, ask for a rework, or describe the changes you want folded in."
                ),
                "header": "Plan approval",
                "multi_select": False,
                "options": [
                    {"label": _PLAN_ACCEPT, "description": "Switch to agent mode and carry out this plan"},
                    {"label": _PLAN_REJECT, "description": "Drop this plan and stop here — nothing is executed"},
                    {"label": _PLAN_REWORK, "description": "Discard this plan and write a new one"},
                ],
            }],
        )
    except Exception:
        return "none", ""
    answers = (resp or {}).get("answers") or []
    answer = answers[0] if answers else {}
    selected = [str(s) for s in (answer.get("selected") or [])]
    feedback = str(answer.get("other_text") or "").strip()
    # Free-text ("Other") always means "revise with this feedback", regardless of selection.
    if feedback:
        return "revise", feedback
    if _PLAN_ACCEPT in selected:
        return "accept", ""
    if _PLAN_REJECT in selected:
        return "reject", ""
    if _PLAN_REWORK in selected:
        return "rework", ""
    return "none", ""


async def _run_plan_mode(
    *,
    agent: Any,
    query: str,
    messages: list[dict],
    execution_context: dict,
    max_steps: int,
    thinking: bool,
    streaming: bool,
    logger: Any,
    cb: dict,
) -> str:
    """Plan-mode loop: gather evidence with read-only tools, then record a plan via the plan/todo tool."""    # Lazy import: agent_loop imports this module (run_agent_query dispatch), so these
    # agent-loop helpers + the tail-called agent loop are fetched at call time.
    from .agent_loop import (
        _advertised_tools, _drain_steer, _run_agent_loop,
        _live_thinking, _live_thinking_budget, _sync_thinking_directive,
        _live_mode, _apply_mode_switch,
    )
    plan_tools = tools_for_plan_mode(_advertised_tools(agent), agent.tool_caps)
    plan_tools = cap_tools_by_relevance(
        plan_tools, query=query, tool_caps=agent.tool_caps, max_tools=max_tools_for(agent.model),
    )
    answer = ""
    plan_recorded = False
    # Both reset whenever the plan goes back to the drawing board (revise / rework).
    post_record_tool_turns = 0
    deliver_nudged = False
    # Advisory evidence gate: a plan recorded with zero exploration is flagged once,
    # never rejected. The query signal behind it is a deliberately broad exit filter
    # (see context.signals) — it fires just as readily for a task that creates
    # something new outside the repo, where no exploration could ever satisfy it.
    # Skipped at enforcement "off": this is discovery babysitting, not a safety guard.
    plan_explore_required = (
        query_requires_repo_discovery(query)
        and resolve_enforcement(agent) != "off"
    )
    plan_evidence_nudged = False
    base_options = {'temperature': 0.2, 'top_k': 25}
    auto_active = getattr(agent, "thinking_depth", None) == THINKING_DEPTH_AUTO

    for plan_nudges in range(max_steps):
        # Pick up mid-run steering (chat-while-busy) before each plan-mode call.
        _drain_steer(agent, messages)

        # The mode is live: leaving plan mid-draft hands the run to the agent loop
        # (which serves both "agent" and read-only "ask"), carrying the conversation
        # and whatever evidence was already gathered with it.
        live_mode = _live_mode(agent, "plan", execution_context)
        if live_mode != "plan":
            system_content = await _apply_mode_switch(agent, messages, new_mode=live_mode)
            return await _run_agent_loop(
                agent=agent,
                query=query,
                active_mode=live_mode,
                messages=messages,
                system_content=system_content,
                execution_context=execution_context,
                max_steps=max_steps,
                thinking=thinking,
                streaming=streaming,
                logger=logger,
                cb=cb,
            )
        # Re-read the reasoning depth so a rung moved mid-plan lands on this call.
        thinking = _live_thinking(agent, thinking)
        auto_active, _ = await _sync_thinking_directive(
            agent, messages, "plan", auto_active, "",
        )
        options = dict(base_options)
        _tb = _live_thinking_budget(agent)
        if thinking and _tb > 0:
            options['thinking_budget'] = _tb
        msg = _stream_chat(
            agent.model,
            messages,
            plan_tools,
            thinking,
            streaming,
            options,
            cancel_flag=getattr(agent, "_cancel_flag", None),
            **cb,
        )
        _process_response(msg, messages, thinking, streamed_thinking=(streaming and cb["think_token_callback"] is not None))
        tool_calls = msg.get("tool_calls") or []
        # Keep whatever prose the model emitted as a fallback answer.
        content = msg.get("content", "")
        if content:
            answer = content

        # Anti-parroting guard: plan recorded, model told to deliver, still calling
        # tools. Past the budget its calls are dropped so the turn falls through to the
        # delivery + approval path. Not conditioned on prose having been emitted — a
        # model stuck here typically emits tool calls and nothing else, and
        # _finalize_answer tolerates an empty answer.
        if tool_calls and plan_recorded:
            post_record_tool_turns += 1
            if post_record_tool_turns > _PLAN_POST_RECORD_TOOL_TURNS:
                _reject_stalled_calls(tool_calls, messages)
                tool_calls = []

        if not tool_calls:
            # Only the final deliver-answer step once the plan document is actually
            # recorded. A plan narrated as chat prose is not the result — nudge and
            # keep looping, so plan mode reliably produces a structured plan.
            if not plan_recorded:
                if plan_nudges <= max_steps - 5:
                    messages.append({"role": "user", "content": PLAN_TODO_NUDGE_EARLY})
                else:
                    messages.append({"role": "user", "content": PLAN_TODO_NUDGE_LATE})
                continue

            # The plan is recorded and has been presented to the user. Ask them to
            # approve it, reject it, or request changes before doing any work.
            decision, feedback = await _request_plan_decision(agent)
            if decision == "accept":
                emit({"type": "status", "text": "  ✔ Plan approved — switching to agent mode"})
                # Persist the switch so the in-session default flips to "agent" and the
                # front-end toggle syncs via the "mode" event. Guarded for
                # non-interactive callers that lack set_mode.
                _set_mode = getattr(agent, "set_mode", None)
                if callable(_set_mode):
                    try:
                        _set_mode("agent")
                    except Exception:
                        pass
                emit({"type": "mode", "mode": "agent"})
                agent_system = await agent._build_system_content(active_mode="agent")
                messages[0]["content"] = agent_system
                messages.append({"role": "user", "content": PLAN_APPROVED_EXECUTE})
                # Seamlessly continue in agent mode, executing the approved plan to
                # completion. _run_agent_loop finalises the answer itself.
                return await _run_agent_loop(
                    agent=agent,
                    query=query,
                    active_mode="agent",
                    messages=messages,
                    system_content=agent_system,
                    execution_context=execution_context,
                    max_steps=max_steps,
                    thinking=thinking,
                    streaming=streaming,
                    logger=logger,
                    cb=cb,
                )
            if decision == "revise":
                emit({"type": "status", "text": "  ↻ Reworking the plan per your feedback"})
                messages.append({"role": "user", "content": plan_revision_nudge(feedback)})
                _clear_recorded_plan(execution_context)
                plan_recorded = False
                post_record_tool_turns = 0
                deliver_nudged = False
                continue
            if decision == "rework":
                emit({"type": "status", "text": "  ↺ Plan sent back — reworking from scratch"})
                messages.append({"role": "user", "content": PLAN_REWORK_NUDGE})
                _clear_recorded_plan(execution_context)
                plan_recorded = False
                post_record_tool_turns = 0
                deliver_nudged = False
                continue
            if decision == "reject":
                # Hard stop: the plan is dropped and nothing is executed. The denial is
                # recorded in history so the next query knows this plan was turned down.
                emit({"type": "status", "text": "  ✗ Plan denied — stopping here"})
                messages.append({"role": "user", "content": PLAN_REJECTED_STOP})
                answer = PLAN_REJECTED_ANSWER
                break
            # "none": no interactive front-end / dismissed — deliver the plan as-is.
            break

        # Safety guard: drop any hallucinated calls to write/execution tools, restrict
        # the dual-use exec tool to read-only discovery, and drop the task-checklist
        # tool (plan mode records the plan document only). Shared with ask mode — see
        # readonly_guard.filter_readonly_tool_calls.
        tool_calls = filter_readonly_tool_calls(
            tool_calls, agent=agent, messages=messages, mode_label="plan",
        )

        await _dispatch_tool_calls(tool_calls, agent, messages, execution_context)
        # Whether a plan exists is the execution context's call, never a second
        # derivation from tool names here: ``observations._observe_todo_flags`` tells
        # the two TASK_PLANNING forms apart by the `plan_steps` arg-role and records
        # `plan_written` (prose document) — the only form plan mode produces.
        if execution_context.get("plan_written"):
            if (plan_explore_required
                    and not plan_evidence_nudged
                    and not has_discovery_evidence(execution_context, min_distinct=DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN)):
                # Advisory, not a rejection: told once, the model either strengthens the
                # plan or says so on delivery. Blocking would discard the plan it just
                # recorded, with nothing guaranteeing it submits that form again.
                plan_evidence_nudged = True
                emit({"type": "status", "text": "  ⚠ Plan not grounded in any exploration — flagged"})
                messages.append({"role": "user", "content": PLAN_EVIDENCE_NUDGE})
            plan_recorded = True

        if not plan_recorded:
            messages.append({"role": "user", "content": (
                PLAN_TODO_NUDGE_EARLY if plan_nudges <= max_steps - 5 else PLAN_TODO_NUDGE_LATE
            )})
        else:
            # Repeating the same deliver nudge verbatim is what the model echoes back;
            # escalate to the firm one once it has been ignored.
            messages.append({"role": "user", "content": (
                PLAN_DELIVER_ANSWER if not deliver_nudged else PLAN_DELIVER_ANSWER_FIRM
            )})
            deliver_nudged = True

    return await _finalize_answer(agent, query, answer, execution_context, messages, logger)

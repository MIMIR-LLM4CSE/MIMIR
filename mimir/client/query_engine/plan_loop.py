"""Plan mode: gather read-only evidence, record a plan, get approval, execute.

``_run_plan_mode`` mirrors the agent loop but with plan tools + the PLAN_BLOCKED/
PLAN_READONLY guards and a two-phase explore/draft gate; on approval it tail-calls
``_run_agent_loop`` (lazy import to avoid the agent_loop↔plan_loop cycle). Extracted
from ``agent_loop.py``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..event_sink import emit
from ..context.execution_context import plan_evidence_ready
from ..context.capabilities import DELEGATE, arg_role, names_with_cap
from ..context.signals import query_requires_repo_discovery
from ..config.models import resolve_enforcement
from ..config.constants import (
    PLAN_EXPLORE_MAX_TURNS, THINKING_DEPTH_AUTO, max_tools_for,
)
from ..guardrails.workflow import (
    PLAN_TODO_NUDGE_EARLY,
    PLAN_TODO_NUDGE_LATE,
    PLAN_DELIVER_ANSWER,
    PLAN_DELIVER_ANSWER_FIRM,
    PLAN_ALREADY_RECORDED_ERROR,
    PLAN_EXPLORE_DELEGATE,
    PLAN_EXPLORE_FIRST,
    PLAN_EXPLORE_BUDGET_SPENT,
    PLAN_APPROVED_EXECUTE,
    PLAN_REWORK_NUDGE,
    PLAN_REJECTED_STOP,
    PLAN_REJECTED_ANSWER,
    plan_revision_nudge,
)
from ..guardrails.nudges import inject_reminder
from ..tool_execution.formatter import normalize_arguments
from .streaming import _DraftHold, _stream_chat, _process_response, _to_dict
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


def _plan_title_arg(name: str, agent: Any) -> str:
    """Name of the argument that titles a plan document, for the tool *name*."""
    roles = arg_role(name, "plan_title", agent.tool_caps)
    return roles[0] if roles else ""


def _plan_document_title(tool_calls: list[Any], agent: Any) -> str:
    """Title carried by the plan-document write in *tool_calls*, if there is one."""
    for tc in tool_calls:
        fn = _to_dict(_to_dict(tc).get("function", {}))
        arg = _plan_title_arg(fn.get("name", ""), agent)
        if not arg:
            continue
        title = str(normalize_arguments(fn.get("arguments") or {}).get(arg) or "").strip()
        if title:
            return title
    return ""


def _pin_plan_title(tool_calls: list[Any], agent: Any, title: str) -> list[Any]:
    """Force *title* onto every plan-document write in *tool_calls*.

    The plan store derives a document's identity from its title, so a revision the
    model re-titles lands in a *second* document: the user keeps reading the draft
    they sent back and two plans compete to be "the" plan. Which document is under
    review is this loop's to decide, not the model's, so the title it settled on is
    reimposed and the revision overwrites the plan in place.
    """
    pinned: list[Any] = []
    for tc in tool_calls:
        tc_d = dict(_to_dict(tc))
        fn = dict(_to_dict(tc_d.get("function", {})))
        arg = _plan_title_arg(fn.get("name", ""), agent)
        args = dict(normalize_arguments(fn.get("arguments") or {})) if arg else {}
        if not arg or str(args.get(arg) or "").strip() == title:
            pinned.append(tc)
            continue
        emit({"type": "status", "text": f"  ↻ Revising the plan in place: {title}"})
        args[arg] = title
        fn["arguments"] = args
        tc_d["function"] = fn
        pinned.append(tc_d)
    return pinned


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

    def _plan_tools(*, exploring: bool) -> list:
        tools = tools_for_plan_mode(_advertised_tools(agent), agent.tool_caps, exploring=exploring)
        return cap_tools_by_relevance(
            tools, query=query, tool_caps=agent.tool_caps, max_tools=max_tools_for(agent.model),
        )

    answer = ""
    plan_recorded = False
    # Title of the plan document under review. Deliberately NOT reset by revise/rework:
    # it is what keeps every revision of this review cycle in the same document.
    plan_title = ""
    # Both reset whenever the plan goes back to the drawing board (revise / rework).
    post_record_tool_turns = 0
    deliver_nudged = False
    # Explore phase: while armed, the plan-document tool is withheld until the model
    # has actually read the code. Offering it from the first turn — under a nudge
    # calling the plan mandatory — is what made a plan *to explore* the cheapest way
    # out; withholding it removes the move instead of policing the plan's wording.
    # Armed by the same query signal as the old advisory gate, a deliberately broad
    # exit filter (see context.signals): it fires just as readily for a task that
    # creates something new outside the repo, where no exploration could satisfy it,
    # hence the turn budget below. Skipped at enforcement "off" — this is discovery
    # babysitting, not a safety guard.
    exploring = (
        query_requires_repo_discovery(query)
        and resolve_enforcement(agent) != "off"
    )
    explore_turns = 0
    plan_tools = _plan_tools(exploring=exploring)

    def _nudge_toward_plan(step: int) -> None:
        """Push toward whatever the current phase owes: evidence, then the document."""
        if exploring:
            text = PLAN_EXPLORE_FIRST
            if names_with_cap(DELEGATE, agent.tool_caps):
                text += PLAN_EXPLORE_DELEGATE
            inject_reminder(messages, text, category="plan_explore", tagged=False)
            return
        inject_reminder(
            messages,
            PLAN_TODO_NUDGE_EARLY if step <= max_steps - 5 else PLAN_TODO_NUDGE_LATE,
            category="plan_todo", tagged=False,
        )

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
        # Phase flip: the plan-document tool appears once the model has actually read
        # the code, or once the explore budget is spent — plan mode must always reach
        # a plan, so thin evidence unlocks it too and the plan states its own gaps.
        if exploring:
            if plan_evidence_ready(execution_context):
                exploring = False
                emit({"type": "status", "text": "  ✔ Exploration complete — drafting the plan"})
            elif explore_turns >= PLAN_EXPLORE_MAX_TURNS:
                exploring = False
                emit({"type": "status", "text": "  ⚠ Exploration thin — unlocking the plan tool"})
                inject_reminder(messages, PLAN_EXPLORE_BUDGET_SPENT, category="plan_explore", tagged=False)
            if exploring:
                explore_turns += 1
            else:
                # Rebuilding the tool list breaks the prompt prefix cache once, the
                # same sanctioned trade as a domain re-arm (see toollist): the phase
                # flips at most once per run, and buys the tool the run now needs.
                plan_tools = _plan_tools(exploring=False)

        # Re-read the reasoning depth so a rung moved mid-plan lands on this call.
        thinking = _live_thinking(agent, thinking)
        auto_active, _ = await _sync_thinking_directive(
            agent, messages, "plan", auto_active, "",
        )
        options = dict(base_options)
        _tb = _live_thinking_budget(agent)
        if thinking and _tb > 0:
            options['thinking_budget'] = _tb
        # Until the plan document exists, a turn that only talks is refused below and
        # nudged back — so its prose is held rather than streamed, instead of showing
        # a plan-shaped answer the loop is about to drop (see _DraftHold).
        hold = (
            _DraftHold(cb["token_callback"])
            if not plan_recorded and cb.get("token_callback") is not None
            else None
        )
        step_cb = {**cb, "token_callback": hold.capture} if hold else cb
        try:
            msg = _stream_chat(
                agent.model,
                messages,
                plan_tools,
                thinking,
                streaming,
                options,
                cancel_flag=getattr(agent, "_cancel_flag", None),
                **step_cb,
            )
        except BaseException:
            # Cancelled or given up on: nothing will refuse this turn any more.
            if hold:
                hold.flush()
            raise
        _process_response(msg, messages, thinking, streamed_thinking=(streaming and cb["think_token_callback"] is not None))
        tool_calls = msg.get("tool_calls") or []
        if hold and tool_calls:
            # The turn acted: its prose is narration above the tool cards, not a
            # plan-shaped answer waiting to be refused.
            hold.flush()
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
                if hold:
                    hold.discard()
                _nudge_toward_plan(plan_nudges)
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
        if plan_title:
            tool_calls = _pin_plan_title(tool_calls, agent, plan_title)

        await _dispatch_tool_calls(tool_calls, agent, messages, execution_context)
        # Whether a plan exists is the execution context's call, never a second
        # derivation from tool names here: ``observations._observe_todo_flags`` tells
        # the two TASK_PLANNING forms apart by the `plan_steps` arg-role and records
        # `plan_written` (prose document) — the only form plan mode produces.
        if execution_context.get("plan_written"):
            # No grounding check here any more: the plan-document tool does not exist
            # until the evidence bar is met, so a plan written over nothing is
            # unreachable rather than flagged after the fact.
            plan_title = plan_title or _plan_document_title(tool_calls, agent)
            plan_recorded = True

        if not plan_recorded:
            _nudge_toward_plan(plan_nudges)
        else:
            # Repeating the same deliver nudge verbatim is what the model echoes back;
            # escalate to the firm one once it has been ignored.
            inject_reminder(
                messages,
                PLAN_DELIVER_ANSWER if not deliver_nudged else PLAN_DELIVER_ANSWER_FIRM,
                category="plan_deliver", tagged=False,
            )
            deliver_nudged = True

    return await _finalize_answer(agent, query, answer, execution_context, messages, logger)

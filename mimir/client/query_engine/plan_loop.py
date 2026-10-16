"""Plan mode: gather read-only evidence, record a plan, get approval, execute.

:class:`PlanLoop` runs the shared step (``loop_base``) with the plan tools, the
read-only guards and a two-phase explore/draft gate. On approval — or when the user
switches mode mid-draft — it hands the turn to the agent loop.
"""
from __future__ import annotations

import asyncio
from typing import Any

from ..event_sink import emit
from ..context.execution_context import plan_evidence_ready
from ..context.capabilities import DELEGATE, arg_role, names_with_cap
from ..context.signals import query_requires_repo_discovery
from ..config.limits import PLAN_EXPLORE_MAX_TURNS
from ..config.models import resolve_enforcement
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
from ..protocol.tool_call import normalize_arguments
from ..protocol.tool_call import as_dict
from .deferral import KIND_PLAN_DECISION, deferred_prompts, mark_deferred
from .loop_base import (
    ModeSwitch, StepLoop, TurnContext, _advertised_tools, _drain_steer,
    _rebuild_system_content,
)
from .readonly_guard import filter_readonly_tool_calls, refusal_reply
from ..guardrails.toollist import tools_for_plan_mode


_PLAN_ACCEPT = "Accept & start"
_PLAN_REJECT = "Reject"
_PLAN_REWORK = "Rework"

# Once the plan is recorded the model owes a prose answer + the approval prompt, but
# some models keep re-reading and re-echoing the plan instead — and the deliver nudge
# never breaks that, since a turn that calls tools never reaches the delivery branch.
# After this many further tool-calling turns its calls are dropped.
_PLAN_POST_RECORD_TOOL_TURNS = 2

# Turns a phase nudge waits before repeating itself after a turn that acted.
_PLAN_NUDGE_EVERY = 5


def _plan_title_arg(name: str, agent: Any) -> str:
    """Name of the argument that titles a plan document, for the tool *name*."""
    roles = arg_role(name, "plan_title", agent.tool_caps)
    return roles[0] if roles else ""


def _plan_document_title(tool_calls: list[Any], agent: Any) -> str:
    """Title carried by the plan-document write in *tool_calls*, if there is one."""
    for tc in tool_calls:
        fn = as_dict(as_dict(tc).get("function", {}))
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
        tc_d = dict(as_dict(tc))
        fn = dict(as_dict(tc_d.get("function", {})))
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
        name = as_dict(as_dict(tc).get("function", {})).get("name", "")
        emit({"type": "status", "text": f"  ⏭ Plan already recorded — skipped '{name}'"})
        messages.append(refusal_reply(tc, PLAN_ALREADY_RECORDED_ERROR))


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


class PlanLoop(StepLoop):
    """Plan mode: read-only evidence, then the plan document, then the user's decision."""

    EMPTY_TURNS_ANSWER = ("The model returned empty turns repeatedly and the plan was "
                          "not written. Nothing was concluded — retry the query.")

    def __init__(self, turn: TurnContext, mode: str, system_content: str, *,
                 resume_decision: bool = False) -> None:
        super().__init__(turn, mode, system_content)
        # The decision a resumed turn comes back to answer (see deferral): the plan was
        # delivered by the turn this one resumes, and only the decision is still open.
        self.resume_decision = resume_decision
        self.answer = ""
        self.plan_recorded = False
        # Title of the plan document under review. Deliberately NOT reset by
        # revise/rework: it is what keeps every revision of this review cycle in the
        # same document.
        self.plan_title = ""
        # Both reset whenever the plan goes back to the drawing board.
        self.post_record_tool_turns = 0
        self.deliver_nudged = False
        # Explore phase: while armed, the plan-document tool is withheld until the
        # model has actually read the code. Offering it from the first turn — under a
        # nudge calling the plan mandatory — is what made a plan *to explore* the
        # cheapest way out; withholding it removes the move instead of policing the
        # plan's wording. Armed by a deliberately broad query signal (see
        # context.signals): it fires just as readily for a task that creates something
        # new outside the repo, where no exploration could satisfy it, hence the turn
        # budget. Skipped at enforcement "off" — this is discovery babysitting, not a
        # safety guard.
        self.exploring = (query_requires_repo_discovery(turn.query)
                          and resolve_enforcement(self.agent) != "off")
        self.explore_turns = 0
        self.tools = self._plan_tools()
        self.last_nudged: dict[str, int] = {}

    def _plan_tools(self) -> list:
        return tools_for_plan_mode(
            _advertised_tools(self.agent), self.agent.tool_caps, exploring=self.exploring,
        )

    def prepare_calls(self, tool_calls: list[Any]) -> list[Any]:
        # Drop hallucinated calls to write/execution tools, keep the dual-use exec tool
        # to read-only discovery, drop the task-checklist tool (plan mode records the
        # plan document only); and keep every revision in the document under review.
        tool_calls = filter_readonly_tool_calls(
            tool_calls, agent=self.agent, messages=self.messages, mode_label="plan",
        )
        if self.plan_title:
            tool_calls = _pin_plan_title(tool_calls, self.agent, self.plan_title)
        return tool_calls

    async def run(self) -> str | ModeSwitch:
        while self.steps_left():
            if self.resume_decision:
                self.resume_decision = False
                self.plan_recorded = True
                outcome = await self.decide()
                if outcome == "continue":
                    self.step += 1
                    continue
                if outcome != "deliver":
                    return outcome
                break

            _drain_steer(self.agent, self.messages)
            # Leaving plan mid-draft hands the run to the agent loop (which serves both
            # "agent" and read-only "ask"), carrying the conversation and whatever
            # evidence was already gathered with it.
            switch = await self.follow_mode()
            if switch is not None:
                return switch
            self.advance_phase()

            await self.refresh_live_settings()
            # Until the plan document exists, a turn that only talks is refused and
            # nudged back — so its prose is held rather than shown as a plan-shaped
            # answer the loop is about to drop.
            msg, hold = await self.call_model(self.tools, may_refuse=not self.plan_recorded)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls and not (msg.get("content") or "").strip():
                gave_up = await self.on_empty_turn(msg, hold)
                if gave_up is not None:
                    return gave_up
                self.step += 1
                continue

            # Whatever prose the model emitted is the fallback answer.
            if msg.get("content"):
                self.answer = msg["content"]

            # Anti-parroting guard: plan recorded, model told to deliver, still calling
            # tools. Past the budget its calls are dropped so the turn falls through to
            # the delivery + approval path.
            if tool_calls and self.plan_recorded:
                self.post_record_tool_turns += 1
                if self.post_record_tool_turns > _PLAN_POST_RECORD_TOOL_TURNS:
                    _reject_stalled_calls(tool_calls, self.messages)
                    tool_calls = []

            if not tool_calls:
                # A plan narrated as chat prose is not the result: nudge and keep
                # looping, so plan mode reliably produces a structured plan.
                if not self.plan_recorded:
                    if hold:
                        hold.discard()
                    self.nudge_toward_plan(forced=True)
                    self.step += 1
                    continue
                # Recorded and presented: the user approves, rejects or sends it back.
                outcome = await self.decide()
                if outcome == "continue":
                    self.step += 1
                    continue
                if outcome != "deliver":
                    return outcome
                break

            ended = await self.dispatch(tool_calls)
            if ended is not None:
                return ended
            # Whether a plan exists is the execution context's call, never a second
            # derivation from tool names here (observations._observe_todo_flags records
            # `plan_written` for the prose document — the only form plan mode produces).
            if self.ec.get("plan_written"):
                self.plan_title = self.plan_title or _plan_document_title(tool_calls, self.agent)
                self.plan_recorded = True
            if not self.plan_recorded:
                self.nudge_toward_plan(forced=False)
            else:
                # Repeating the same deliver nudge verbatim is what the model echoes
                # back; escalate to the firm one once it has been ignored.
                inject_reminder(
                    self.messages,
                    PLAN_DELIVER_ANSWER if not self.deliver_nudged else PLAN_DELIVER_ANSWER_FIRM,
                    category="plan_deliver", tagged=False,
                    execution_context=self.ec, step=self.step,
                )
                self.deliver_nudged = True
            self.step += 1

        return await self.finalize(self.answer)

    def advance_phase(self) -> None:
        """Flip from exploring to drafting once the evidence is in, or the budget spent.

        Plan mode must always reach a plan, so thin evidence unlocks the tool too and
        the plan states its own gaps. Rebuilding the tool list breaks the prompt prefix
        once — the phase flips at most once per run, and buys the tool the run needs.
        """
        if not self.exploring:
            return
        if plan_evidence_ready(self.ec):
            self.exploring = False
            emit({"type": "status", "text": "  ✔ Exploration complete — drafting the plan"})
        elif self.explore_turns >= PLAN_EXPLORE_MAX_TURNS:
            self.exploring = False
            emit({"type": "status", "text": "  ⚠ Exploration thin — unlocking the plan tool"})
            inject_reminder(self.messages, PLAN_EXPLORE_BUDGET_SPENT, category="plan_explore",
                            tagged=False, execution_context=self.ec, step=self.step)
        if self.exploring:
            self.explore_turns += 1
        else:
            self.tools = self._plan_tools()

    def nudge_toward_plan(self, *, forced: bool) -> None:
        """Push toward whatever the current phase owes: evidence, then the document.

        Throttled after a turn that acted: fired from every tool turn, the identical
        text landed nine times in one observed run — and a user turn arriving between
        every single tool result is itself what teaches one-call-per-turn. *forced* is
        the turn that only talked: that prose is discarded, so the reminder is the sole
        thing telling the model why, and skipping it would loop the same refusal.
        """
        category = "plan_explore" if self.exploring else "plan_todo"
        if not forced:
            previous = self.last_nudged.get(category)
            if previous is not None and self.step - previous < _PLAN_NUDGE_EVERY:
                return
        self.last_nudged[category] = self.step
        if self.exploring:
            text = PLAN_EXPLORE_FIRST
            if names_with_cap(DELEGATE, self.agent.tool_caps):
                text += PLAN_EXPLORE_DELEGATE
            inject_reminder(self.messages, text, category="plan_explore", tagged=False,
                            execution_context=self.ec, step=self.step)
            return
        late = not self.unbounded and self.step > self.turn.max_steps - 5
        inject_reminder(self.messages, PLAN_TODO_NUDGE_LATE if late else PLAN_TODO_NUDGE_EARLY,
                        category="plan_todo", tagged=False,
                        execution_context=self.ec, step=self.step)

    def back_to_drafting(self) -> None:
        """Forget the recorded plan: a reworked one has to be written afresh."""
        _clear_recorded_plan(self.ec)
        self.plan_recorded = False
        self.post_record_tool_turns = 0
        self.deliver_nudged = False

    async def decide(self) -> str | ModeSwitch:
        """Ask for the plan decision and act on it.

        Returns ``"continue"`` (back to planning), ``"deliver"`` (the plan as it
        stands is the answer), or a :class:`ModeSwitch` to the agent loop to execute
        the approved plan.
        """
        decision, feedback = await _request_plan_decision(self.agent)
        # The user left while the question was up: the plan stands delivered, and the
        # decision is asked again when they come back (query_engine.deferral).
        if any(not d.get("call_id") for d in deferred_prompts(self.agent)):
            mark_deferred(self.agent, kind=KIND_PLAN_DECISION, query=self.turn.query,
                          mode="plan")
            return "deliver"
        if decision == "accept":
            emit({"type": "status", "text": "  ✔ Plan approved — switching to agent mode"})
            # Persisted so the in-session default flips to "agent" and the front-end
            # toggle syncs via the "mode" event. Guarded for callers without set_mode.
            set_mode = getattr(self.agent, "set_mode", None)
            if callable(set_mode):
                try:
                    set_mode("agent")
                except Exception:
                    pass
            emit({"type": "mode", "mode": "agent"})
            # Through the shared rule: the agent-mode prompt renders the checklist this
            # plan just wrote, and the skill block survives the handoff.
            agent_system = await _rebuild_system_content(self.agent, "agent", self.ec)
            self.messages[0]["content"] = agent_system
            self.messages.append({"role": "user", "content": PLAN_APPROVED_EXECUTE})
            return ModeSwitch("agent", agent_system)
        if decision == "revise":
            emit({"type": "status", "text": "  ↻ Reworking the plan per your feedback"})
            self.messages.append({"role": "user", "content": plan_revision_nudge(feedback)})
            self.back_to_drafting()
            return "continue"
        if decision == "rework":
            emit({"type": "status", "text": "  ↺ Plan sent back — reworking from scratch"})
            self.messages.append({"role": "user", "content": PLAN_REWORK_NUDGE})
            self.back_to_drafting()
            return "continue"
        if decision == "reject":
            # Hard stop: the plan is dropped and nothing is executed. The denial is
            # recorded in history so the next query knows this plan was turned down.
            emit({"type": "status", "text": "  ✗ Plan denied — stopping here"})
            self.messages.append({"role": "user", "content": PLAN_REJECTED_STOP})
            self.answer = PLAN_REJECTED_ANSWER
            return "deliver"
        # "none": no interactive front-end / dismissed — deliver the plan as-is.
        return "deliver"

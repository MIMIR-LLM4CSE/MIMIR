"""The agent loop: agent mode, and the read-only ask mode.

Each step calls the model with the mode's tools and either dispatches the calls it
made or judges the answer it gave — a pending nudge, the evidence handback or an empty
turn send it back to work; otherwise the answer is finalised. Plan mode is a loop of
its own (``plan_loop``); switching to it mid-run hands the turn over.
"""
from __future__ import annotations

import os
from typing import Any

from ..config.limits import NUDGE_MAX_CONSECUTIVE_NOOP
from ..config.models import READONLY_MODES
from ..event_sink import emit
from ..guardrails.builtin_check import sweep_builtin_checks
from ..guardrails.nudges import (
    inject_reminder,
    maybe_append_nudge,
    needs_incomplete_finalization,
    nudge_pending,
)
from ..guardrails.workflow import (
    STEP_LIMIT_NUDGE,
    TERMINATION_STEP_LIMIT,
    evidence_handback_message,
    finalize_incomplete_answer,
)
from .loop_base import (
    ModeSwitch, StepLoop, TurnContext, _advertised_tools, _drain_steer,
    _rebuild_system_content,
)
from .toollist import tools_for_context, tools_for_readonly_mode


def _mode_tools(
    agent: Any, query: str, execution_context: dict, active_mode: str,
) -> list[dict]:
    """Build the tool list for *active_mode*: read-only filter, then context pruning.

    The single place the mode's tool surface is decided, so the initial build, a
    domain re-arm, and a mid-run mode switch can never disagree about it.
    """
    tools = _mode_advertised_tools(agent, active_mode)
    return tools_for_context(
        query=query,
        execution_context=execution_context,
        tools=tools,
        tool_caps=agent.tool_caps,
    )


def _mode_advertised_tools(agent: Any, active_mode: str) -> list[dict]:
    """The advertised set narrowed by the mode, from the agent when it can do it.

    The rule lives on the agent (``advertised_tools_for_mode``) so the context bar
    measures the same list this builds; the fallback keeps the loop's lightweight
    test stubs, which carry neither method, working as before.
    """
    fn = getattr(agent, "advertised_tools_for_mode", None)
    if callable(fn):
        return fn(active_mode)
    tools = _advertised_tools(agent)
    if active_mode in READONLY_MODES:
        tools = tools_for_readonly_mode(tools, agent.tool_caps, mode=active_mode)
    return tools


async def _sync_checklist(
    agent: Any,
    messages: list[dict],
    active_mode: str,
    system_content: str,
    execution_context: dict,
) -> str:
    """Rewrite the system message when the on-disk task checklist has changed.

    THE INVARIANT: a block of STATE never occupies the last position of the prompt.

    Every chat template appends its generation prompt after the last message, so
    whatever sits there is what the model is being asked to respond to or continue. A
    checklist has nothing to ask and nothing to answer — putting it last hands the model
    a status report where its turn should be. What that produces is template-dependent
    and the failure is not: one template continued the block's own text until the step
    budget ran out, another emitted a short reasoning block and EOS. Two symptoms, one
    cause. Measured on one backend (37 empty turns / 108 draws with the block in the
    tail, 0 / 84 without, 0/40 with the same text in messages[0]); the numbers say where
    it was quantified, not where it applies.

    The corollary is the sorting rule this loop follows everywhere: PILOTAGE — nudges,
    reminders, the empty-turn retry — belongs in the last position, because asking for
    the next turn IS its function, and it measured harmless there. STATE goes in
    messages[0]. Nothing here tests the model or the template: the placement is
    unconditional, which is the point: a per-model workaround here — a tail ``user`` turn
    rather than a tail ``system`` one, because a tail ``system`` turn breaks one template's
    generation prompt — is a distinction that measures irrelevant to the real failure
    (7/40 vs 8/40) and has to be maintained for ever.

    Two gates, cheapest first. The trigger is the file's mtime rather than the
    ``todo_update`` tool because the tool is not its only writer (``todo_write``, a
    plain write to the path, a sub-agent), and one stat() per step costs less than a
    stale checklist. The content comparison after the rebuild is not a
    micro-optimisation: the todo server re-saves the file even when the text does not
    change (ticking an item already ticked), and without it every such touch would cost
    the whole prefix rather than nothing.

    Returns the system content to keep budgeting against, like
    :func:`loop_base._sync_thinking_directive`.
    """
    todo_fp = execution_context.get("todo_file_path", "")
    if not todo_fp:
        return system_content
    try:
        stamp = os.stat(todo_fp).st_mtime_ns
    except OSError:
        return system_content  # no checklist on disk — nothing to refresh
    if stamp == execution_context.get("_checklist_stamp"):
        return system_content
    execution_context["_checklist_stamp"] = stamp
    if not (getattr(agent, "_build_system_content", None) and messages
             and messages[0].get("role") == "system"):
        return system_content
    rebuilt = await _rebuild_system_content(agent, active_mode, execution_context)
    if rebuilt == messages[0].get("content"):
        return system_content  # touched, not changed: the prefix still hits
    messages[0]["content"] = rebuilt
    emit({"type": "status", "text": "  ↻ Task checklist changed — system prompt refreshed"})
    return rebuilt


def _turn_may_be_rejected(
    agent: Any, query: str, active_mode: str, execution_context: dict,
) -> bool:
    """True if a bare final turn produced *now* would be sent back to the model.

    Mirrors, before the call, the two post-call branches below that refuse an
    answer: a pending nudge (subject to the same no-op streak cap) and the
    once-per-query evidence handback.
    """
    sweep_builtin_checks(execution_context)
    prospective_noop = execution_context.get("consecutive_noop_turns", 0) + 1
    if prospective_noop <= NUDGE_MAX_CONSECUTIVE_NOOP and nudge_pending(
        agent=agent,
        query=query,
        active_mode=active_mode,
        execution_context=execution_context,
    ):
        return True
    return needs_incomplete_finalization(execution_context) and not execution_context.get(
        "evidence_handback_used"
    )


class AgentLoop(StepLoop):
    """Agent mode, and ask mode — the same loop with the write/exec tools stripped."""

    def __init__(self, turn: TurnContext, mode: str, system_content: str) -> None:
        super().__init__(turn, mode, system_content)
        # Computed ONCE and reused every step. The list is query-stable (pruning
        # depends only on the query, not on the evolving execution context), so
        # recomputing it per step only churned the prompt prefix and broke prefix
        # caching. A read-only mode's write/exec tools are stripped up front.
        self.tools = _mode_tools(self.agent, turn.query, self.ec, mode)

    def serves(self, mode: str) -> bool:
        # Plan mode is a different loop shape (evidence → plan → approval).
        return mode != "plan"

    def on_mode_changed(self) -> None:
        self.tools = _mode_tools(self.agent, self.turn.query, self.ec, self.mode)

    def thinking_budget(self, base: int) -> int:
        """Scaled by workflow phase: the full rung while reasoning about code, less
        once the work is being validated or concluded."""
        phase = self.ec.get("workflow_state", "discover")
        if phase in ("discover", "edit"):
            return base
        if phase == "validate":
            return max(1024, base // 2)
        return max(512, base // 4)

    async def refresh_live_settings(self) -> None:
        await super().refresh_live_settings()
        # Same reason, same place: a checklist the model ticked off last step is only
        # visible to it once messages[0] carries it, and the budget must account for
        # the message it will actually send.
        self.system_content = await _sync_checklist(
            self.agent, self.messages, self.mode, self.system_content, self.ec,
        )

    async def run(self) -> str | ModeSwitch:
        budget = self.turn.max_steps
        while self.steps_left():
            _drain_steer(self.agent, self.messages)
            switch = await self.follow_mode()
            if switch is not None:
                return switch

            # Two steps before the budget boundary, ask for a summary of what is done
            # and what remains, so the handoff is meaningful rather than a bare "limit".
            if budget > 0 and budget - 3 <= self.step < budget - 1:
                inject_reminder(self.messages, STEP_LIMIT_NUDGE, category="step_limit",
                                tagged=False, execution_context=self.ec, step=self.step)
            # Steps since the last successful edit, so the nudges can tell a refactor
            # in progress from a paused one.
            self.ec["steps_since_last_edit"] = self.ec.get("steps_since_last_edit", 0) + 1

            await self.refresh_live_settings()
            msg, hold = await self.call_model(
                self.tools,
                may_refuse=_turn_may_be_rejected(
                    self.agent, self.turn.query, self.mode, self.ec),
            )
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                answer = await self.on_answer(msg, hold)
                if answer is not None:
                    return answer
                self.step += 1
                continue

            # The model acted — the no-op streak resets, so it keeps earning reminders
            # as long as it keeps making progress between them.
            self.ec["consecutive_noop_turns"] = 0
            ended = await self.dispatch(tool_calls)
            if ended is not None:
                return ended
            # Trim/compact again after the results so history stays bounded between
            # steps; the enforcement before each call is what guarantees the fit.
            self.enforce_budget(self.tools)
            self.step += 1

        # Only a caller-supplied budget running out gets here; every other exit (a
        # final answer, a guard) returns from inside the loop. A run that ran out of
        # steps is unfinished whatever else the ledger says.
        answer = "Reached the maximum number of steps without a final answer."
        sweep_builtin_checks(self.ec)
        if needs_incomplete_finalization(self.ec):
            answer = finalize_incomplete_answer(answer, self.ec, TERMINATION_STEP_LIMIT)
        answer = await self.finalize(answer)
        return _review_batch(self.agent, answer, self.messages)

    async def on_answer(self, msg: dict, hold: Any) -> str | None:
        """A turn with no tool call: the final answer, or a reason to go on.

        Returns the finalised answer, or None when the model is sent back to work.
        """
        answer = msg.get("content", "")
        if not (answer or "").strip():
            return await self.on_empty_turn(msg, hold)
        # A nudge is only worth sending while the model still *acts* on reminders:
        # once it answers one with another bare turn, it is ignoring them.
        noop_turns = self.ec.get("consecutive_noop_turns", 0) + 1
        self.ec["consecutive_noop_turns"] = noop_turns
        # The mandatory check happens here rather than after each write: the model has
        # stopped editing, so every dirty file is at the revision it will ship at.
        sweep_builtin_checks(self.ec)
        if noop_turns <= NUDGE_MAX_CONSECUTIVE_NOOP and maybe_append_nudge(
            agent=self.agent, query=self.turn.query, active_mode=self.mode,
            execution_context=self.ec, messages=self.messages,
        ):
            if hold:
                hold.discard()
            return None
        if needs_incomplete_finalization(self.ec):
            # Once per query, before the report is assembled: the model has never seen
            # the ledger its summary is about to contradict.
            if not self.ec.get("evidence_handback_used"):
                self.ec["evidence_handback_used"] = True
                inject_reminder(self.messages, evidence_handback_message(self.ec),
                                category="evidence_handback", tagged=False,
                                execution_context=self.ec, step=self.step)
                if hold:
                    hold.discard()
                return None
            answer = finalize_incomplete_answer(answer, self.ec)
        answer = await self.finalize(answer)
        return _review_batch(self.agent, answer, self.messages)


def _review_batch(agent: Any, answer: str, messages: list[dict]) -> str:
    """Put the turn's batched edits to the user, and say on the answer what became of them.

    Both ways out of the loop end here — an answer, and running out of steps — so a
    batch is never left unreviewed, and never reviewed without the outcome being told.
    """
    approvals = getattr(agent, "approvals", None)
    if not approvals or not getattr(approvals, "_pending_review", None):
        return answer
    confirm_msg = "Changes applied." if approvals.flush_pending_review() else "Changes reverted."
    messages.append({"role": "assistant", "content": confirm_msg})
    return answer + "\n" + confirm_msg

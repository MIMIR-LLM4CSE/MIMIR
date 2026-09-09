from __future__ import annotations

import os
from typing import Any

from . import streaming as _streaming
from ..event_sink import emit
from .toollist import tools_for_context, tools_for_readonly_mode
from .streaming import _DraftHold, _note_truncated_turn, _process_response, _stream_chat
from .history import _enforce_context_budget, reconcile_tool_pairs
from .finalize import _finalize_answer
from .dispatch import _dispatch_tool_calls, _post_dispatch_inject
from .readonly_guard import filter_readonly_tool_calls
from .plan_loop import _run_plan_mode
from ..config.constants import (
    MAX_AGENT_STEPS,
    AGENT_STEP_SOFT_BUDGET,
    AGENT_STEP_EXTENSION,
    AGENT_STEP_HARD_CEILING,
    AGENT_EMPTY_TURN_RETRIES,
    NUDGE_MAX_CONSECUTIVE_NOOP,
    THINKING_DEPTH_AUTO,
)
from ..config.models import READONLY_MODES, VALID_MODES
from ..context import validate_execution_context
from ..guardrails.builtin_check import sweep_builtin_checks
from ..guardrails.workflow import (
    evidence_handback_message,
    finalize_incomplete_answer,
    EMPTY_TURN_OPENING,
    empty_turn_retry_message,
    STEP_LIMIT_NUDGE,
    TERMINATION_STEP_LIMIT,
    TERMINATION_USER_STOPPED,
)
from ..guardrails.nudges import (
    drop_transient_reminders,
    inject_reminder,
    maybe_append_nudge,
    needs_incomplete_finalization,
    nudge_pending,
)


def _live_thinking(agent: Any, fallback: bool) -> bool:
    """The thinking flag to use for the *next* model call, re-read from the agent.

    The user can move the depth rung while a query is already running (the WS
    command handler and the CLI mutate the agent from outside the loop), and the
    change has to land on the next step rather than the next query. Callers that
    drive a bare stub without the attribute (tests, embedders) keep the value they
    passed in.
    """
    val = getattr(agent, "thinking", None)
    return fallback if val is None else bool(val)


def _live_thinking_budget(agent: Any) -> int:
    """The token budget for the next model call. ``-1`` = unbudgeted."""
    val = getattr(agent, "thinking_budget", -1)
    return val if isinstance(val, int) else -1


_OBSERVED_MODE = "_observed_agent_mode"


def _live_mode(agent: Any, active_mode: str, execution_context: dict) -> str:
    """The mode to run the *next* step in, re-read from the agent.

    The mode is a live setting, not a per-query constant: the user can flip it
    while a query is already running (the WS command handler mutates the agent
    from outside the loop), and the change has to land on the next step rather
    than the next query.

    What counts is a *change* to ``agent.mode``, not its value — a caller may run
    one query in an explicit mode without touching the agent's own (sub-agents and
    the runner do), and that must not read as the user switching. So the last
    observed value is kept in the execution context, seeded at query start, and the
    change is consumed here. Returns *active_mode* unchanged when nothing moved.
    """
    val = getattr(agent, "mode", None)
    if not isinstance(val, str):
        return active_mode
    val = val.strip().lower()
    if val not in VALID_MODES or val == execution_context.get(_OBSERVED_MODE):
        return active_mode
    execution_context[_OBSERVED_MODE] = val
    return val


async def _rebuild_system_content(
    agent: Any, active_mode: str, execution_context: dict,
) -> str:
    """The system message as it must be after ANY rebuild: prompt + skill block.

    The skill block is folded into messages[0] rather than appended as a second
    ``system`` message (appending made it accumulate across queries — one session
    carried nine copies). The cost of that choice is that every rebuild has to put it
    back, and four sites rebuilt messages[0] with three different answers to that
    question: a mode switch, a thinking-rung change, the plan→agent handoff and the
    checklist refresh. The three that forgot dropped the skill silently, mid-run, on a
    user action. One function, so the rule cannot be half-applied.
    """
    content = await agent._build_system_content(active_mode=active_mode)
    return content + (execution_context.get("_skill_suffix") or "")


async def _sync_thinking_directive(
    agent: Any,
    messages: list[dict],
    active_mode: str,
    auto_active: bool,
    system_content: str,
    execution_context: dict,
) -> tuple[bool, str]:
    """Rewrite the system message when the run enters or leaves the "auto" rung.

    Everything else about the depth travels in the backend payload and is picked up
    per step for free, but the "auto" rung also carries a calibration directive in
    the system prompt — invisible to the model until messages[0] is rebuilt. Only a
    change of rung pays for it (one prefix-cache miss, on an explicit user action);
    steady state is a single comparison. Returns the new auto-ness and the system
    content to keep budgeting against.
    """
    depth = getattr(agent, "thinking_depth", None)
    if depth is None:
        return auto_active, system_content
    now = depth == THINKING_DEPTH_AUTO
    if now == auto_active:
        return auto_active, system_content
    if getattr(agent, "_build_system_content", None) and messages \
            and messages[0].get("role") == "system":
        system_content = await _rebuild_system_content(
            agent, active_mode, execution_context,
        )
        messages[0]["content"] = system_content
    return now, system_content


def _mode_tools(
    agent: Any, query: str, execution_context: dict, active_mode: str,
) -> list[dict]:
    """Build the tool list for *active_mode*: read-only filter, then context pruning.

    The single place the mode's tool surface is decided, so the initial build, a
    domain re-arm, and a mid-run mode switch can never disagree about it.
    """
    tools = _advertised_tools(agent)
    if active_mode in READONLY_MODES:
        tools = tools_for_readonly_mode(tools, agent.tool_caps, mode=active_mode)
    return tools_for_context(
        query=query,
        execution_context=execution_context,
        tools=tools,
        tool_caps=agent.tool_caps,
    )


async def _apply_mode_switch(
    agent: Any, messages: list[dict], *, new_mode: str, execution_context: dict,
) -> str:
    """Rebuild the mode-dependent system prompt after a mid-run mode change.

    The mode decides whole sections of the prompt (the task checklist, the PLAN
    directive, the ASK directive), so messages[0] has to be rewritten for the
    model to act on the new mode at all. That costs the prefix cache for the rest
    of the query — a deliberate, user-triggered break, so it is reported rather
    than hidden, like the domain re-arm. Returns the new system content.
    """
    emit({"type": "status", "text": f"  ↻ Switched to {new_mode} mode mid-run"})
    emit({"type": "mode", "mode": new_mode})
    system_content = await _rebuild_system_content(agent, new_mode, execution_context)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_content
    return system_content


def _advertised_tools(agent: Any) -> list[dict]:
    """The tool list to expose this step: the agent's soft-hide-filtered set when it
    provides one (``advertised_tools``), else its raw ``tools`` (keeps lightweight
    test stubs working, matching the loop's defensive ``getattr`` idiom)."""
    fn = getattr(agent, "advertised_tools", None)
    return fn() if callable(fn) else agent.tools


# Caps for the tool-result `output` surfaced in the UI (expandable tool rows).
# Kept small so structured events don't bloat the WebSocket stream.


def _note_empty_turn(msg: dict, messages: list[dict], execution_context: dict,
                     step: int) -> None:
    """Record the shape of the prompt that produced an empty turn.

    An empty turn is the one failure the loop cannot explain from what it keeps: the
    message itself is empty by definition, and the transient pin and reminders are
    gone from history by the time anyone reads the session back. The first empty turn
    of the session that prompted this arrived on a perfectly ordinary prompt, so the
    trigger is not in the message list's shape alone — which is exactly why the shape
    has to be recorded at the moment it happens rather than reconstructed later.

    Names the reminder categories injected for THIS call: the transcript already says
    a reminder fired and the loop already says a turn came back empty, but nothing
    tied the two together, so "a reminder — or one category of reminder — is what
    empties the turn" could not be tested against a real run.
    """
    tail = " → ".join(str(m.get("role")) for m in messages[-6:])
    # Read after the fact: the reminders that were in this prompt have already been
    # taken back out by the time the turn is judged empty.
    cats = list(execution_context.get("_last_call_reminders") or [])
    emit({"type": "status", "text": (
        f"  ⓘ Empty turn diagnostics — step {step}, {len(messages)} messages, "
        f"finish_reason={msg.get('finish_reason') or 'none'}, "
        f"reminders in prompt={len(cats)}"
        + (f" ({', '.join(sorted(set(cats)))})" if cats else "")
        + f", tail: {tail}"
    )})


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
    unconditional, which is the point. It replaces a per-model workaround — the block
    used to be a tail ``user`` turn specifically because a tail ``system`` turn broke
    one template's generation prompt, a distinction that measured irrelevant to the
    real failure (7/40 vs 8/40) and that no longer has to be maintained.

    Two gates, cheapest first. The trigger is the file's mtime rather than the
    ``todo_update`` tool because the tool is not its only writer (``todo_write``, a
    plain write to the path, a sub-agent), and one stat() per step costs less than a
    stale checklist. The content comparison after the rebuild is not a
    micro-optimisation: the todo server re-saves the file even when the text does not
    change (ticking an item already ticked), and without it every such touch would cost
    the whole prefix rather than nothing.

    Returns the system content to keep budgeting against, like
    :func:`_sync_thinking_directive`.
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


def _drain_steer(agent: Any, messages: list[dict]) -> None:
    """Inject any user "steer" messages queued mid-run as user turns at a step boundary.

    Front-ends that support chatting-while-busy (the WebSocket server) patch a
    ``_poll_steer`` callback onto the agent that returns and clears any messages the
    user typed while the agent was working. Called at the top of each loop step — after
    the previous step's tool results are already in ``messages`` and before the next
    model call — so the steer reads as the next user turn and the model can adjust
    course without discarding in-flight work.

    Optional by design: callers that never set ``_poll_steer`` (CLI, sub-agents, tests)
    are unaffected, mirroring the ``getattr(agent, "_cancel_flag", None)`` pattern. The
    injected messages persist in history. Adjacent user turns (e.g. following a
    post-dispatch nudge) are reconciled downstream by the backend's
    consecutive-user-message merge, so no folding is needed here.

    Nothing may be appended after a steer before the call: it is a real user turn, and
    the last position is what the model answers. The checklist used to be appended
    there — after this, just before the call — so a mid-run instruction was merged into
    one user turn ending in a status block, and the model answered the block. That is
    the same defect as :func:`_sync_checklist` documents, in its most visible form.
    """
    poll = getattr(agent, "_poll_steer", None)
    if not poll:
        return
    for text in (poll() or []):
        text = (text or "").strip()
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        emit({"type": "steer_injected", "text": text})


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


def _checkpoint_summary(messages: list[dict], execution_context: dict, step: int) -> str:
    """Build a short progress blurb shown to the user at a soft-budget checkpoint."""
    last_assistant = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant = str(m["content"]).strip()
            break
    parts = [f"Reached {step} steps."]
    dirty = sorted(execution_context.get("dirty_written_files", set()) or set())
    if dirty:
        parts.append("Modified: " + ", ".join(os.path.basename(p) for p in dirty[:8]))
    if last_assistant:
        parts.append(last_assistant[:400])
    return "  ".join(parts)


# Plan-approval choice labels. Shared with the user-question prompt so the loop
# can map the selection back to an action.


async def _run_agent_loop(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    messages: list[dict],
    system_content: str,
    execution_context: dict,
    max_steps: int,
    thinking: bool,
    streaming: bool,
    logger: Any,
    cb: dict,
) -> str:
    """Agent-mode step loop with policy guardrails, nudges, and history budgeting."""
    step = 0
    # Interactive front-ends (CLI, WebSocket) let the user extend a long run.
    # The loop runs up to a soft budget, then asks whether to continue, granting
    # another extension block each time up to a hard ceiling. Non-interactive
    # callers (sub-agents, tests) keep a fixed budget == ceiling.
    interactive = bool(getattr(agent, "allow_continue_prompt", False))
    if interactive:
        budget = AGENT_STEP_SOFT_BUDGET
        hard = AGENT_STEP_HARD_CEILING
    else:
        budget = max_steps
        hard = max_steps
    options = {'temperature': 0.3}
    # A sub-agent caps its own answer: left to the backend default it gets the whole
    # answer reserve (tens of thousands of tokens), and a step that runs away is
    # invisible from outside — one mute row for as long as it takes to generate.
    _answer_cap = int(getattr(agent, "max_answer_tokens", 0) or 0)
    if _answer_cap > 0:
        options['max_tokens'] = _answer_cap
    # Whether the system message currently carries the "auto" calibration directive.
    # Tracked so a mid-run rung change can rewrite it (see _sync_thinking_directive).
    auto_active = getattr(agent, "thinking_depth", None) == THINKING_DEPTH_AUTO

    # Loop-invariant setup, lifted out of the per-step body.
    _tok = lambda text: _streaming.get_backend().count_text_tokens(agent.model, text)  # noqa: E731
    compact_fn = getattr(agent, "compact_messages", None)
    context_mode = getattr(agent, "context_mode", "full")
    # Compute the per-query tool list ONCE and reuse it every step. The list is
    # query-stable (pruning + relevance cap depend only on the query, not on the
    # evolving execution_context), so recomputing per step only churned the prompt
    # prefix and broke vLLM prefix caching. The former discover-gated *hiding* of
    # write tools now lives in the call-time policy (check_write_policy), keeping
    # this list stable.
    # A read-only mode (ask) runs this same loop, so its write/exec tools are
    # stripped up front — before the context filter, which only prunes by relevance.
    readonly = active_mode in READONLY_MODES
    query_tools = _mode_tools(agent, query, execution_context, active_mode)
    # Consecutive turns that returned neither prose nor a tool call (see the guard
    # in the loop body).
    empty_turns = 0
    # Only the two exits below reach the post-loop report; a final answer returns
    # from inside the loop.
    termination = TERMINATION_STEP_LIMIT

    while step < hard:
        # Pick up anything the user typed mid-run (chat-while-busy steering) and
        # inject it as the next user turn before this step's model call.
        _drain_steer(agent, messages)

        # The mode is live: the user can flip it while this query is running, and
        # the change must land on THIS step. Plan mode is a different loop shape
        # (evidence → checklist → approval), so switching into it hands the run
        # over rather than trying to emulate it here.
        live_mode = _live_mode(agent, active_mode, execution_context)
        if live_mode != active_mode:
            active_mode = live_mode
            readonly = active_mode in READONLY_MODES
            system_content = await _apply_mode_switch(
                agent, messages, new_mode=active_mode,
                execution_context=execution_context,
            )
            if active_mode == "plan":
                return await _run_plan_mode(
                    agent=agent,
                    query=query,
                    messages=messages,
                    execution_context=execution_context,
                    max_steps=max_steps,
                    thinking=thinking,
                    streaming=streaming,
                    logger=logger,
                    cb=cb,
                )
            query_tools = _mode_tools(agent, query, execution_context, active_mode)

        # Two steps before the current budget boundary, nudge the model to
        # summarise what's done and what remains so the handoff (user checkpoint
        # or final answer) is meaningful rather than a bare "reached limit".
        if budget - 3 <= step < budget - 1:
            inject_reminder(messages, STEP_LIMIT_NUDGE, category="step_limit", tagged=False,
                            execution_context=execution_context, step=step)

        # Track how many steps have elapsed since the last successful edit,
        # so nudge_logic can distinguish mid-refactor from a paused state.
        execution_context["steps_since_last_edit"] = (
            execution_context.get("steps_since_last_edit", 0) + 1
        )

        # Re-read the reasoning depth: the user may have moved the rung since the
        # last step, and the change must land on THIS call. Done before the budget
        # check so the (possibly rebuilt) system message is what gets accounted for.
        thinking = _live_thinking(agent, thinking)
        auto_active, system_content = await _sync_thinking_directive(
            agent, messages, active_mode, auto_active, system_content,
            execution_context,
        )
        # Same reason, same place: a checklist the model ticked off last step is only
        # visible to it once messages[0] carries it, and the budget below must account
        # for the message it will actually send.
        system_content = await _sync_checklist(
            agent, messages, active_mode, system_content, execution_context,
        )

        # Enforce the context budget BEFORE every LLM call (not only after tool
        # dispatch) so the first iteration — and any call whose history grew via
        # injected nudges — can never overflow the model window. Accounts for the
        # (stable) tools schema sent alongside `messages`.
        _enforce_context_budget(
            messages, system_content, query_tools, execution_context,
            agent.model, context_mode, compact_fn, _tok,
        )

        # Adaptive thinking budget: scale based on workflow phase.
        # Discovery/edit phases get a larger budget; validate/conclude get less.
        # No-op on the unbudgeted rungs ("auto"/"max"), where depth is the model's.
        step_options = dict(options)
        _base_tb = _live_thinking_budget(agent)
        if thinking and _base_tb > 0:
            _wf = execution_context.get("workflow_state", "discover")
            if _wf in ("discover", "edit"):
                _scaled_tb = _base_tb  # full budget while reasoning about code
            elif _wf == "validate":
                _scaled_tb = max(1024, _base_tb // 2)
            else:  # conclude and anything else
                _scaled_tb = max(512, _base_tb // 4)
            step_options['thinking_budget'] = _scaled_tb

        # Hold this turn's prose off the screen when the loop still has grounds to
        # refuse it (see _DraftHold): rendering a turn that a nudge then discards is
        # what makes an answer appear and vanish. Turns with nothing pending stream
        # straight through, which is the common case.
        hold = (
            _DraftHold(cb["token_callback"])
            if cb.get("token_callback") is not None
            and _turn_may_be_rejected(agent, query, active_mode, execution_context)
            else None
        )
        step_cb = {**cb, "token_callback": hold.capture} if hold else cb

        try:
            msg = _stream_chat(
                agent.model,
                messages,
                query_tools,
                thinking,
                streaming,
                step_options,
                cancel_flag=getattr(agent, "_cancel_flag", None),
                **step_cb,
            )
        except BaseException:
            # Cancelled, or the backend gave up: no guardrail will get to refuse
            # this turn, so whatever prose was held is the user's to keep.
            if hold:
                hold.flush()
            raise
        finally:
            # The reminders injected for THIS call have now been put to the model.
            # Keeping them is what let one sentence reach 21 identical copies in a
            # single session; the emitted nudge_injected events keep the diagnosis.
            drop_transient_reminders(messages, execution_context)
        _process_response(msg, messages, thinking, streamed_thinking=(streaming and cb["think_token_callback"] is not None))
        _note_truncated_turn(msg, step)

        tool_calls = msg.get("tool_calls") or []
        if hold and tool_calls:
            # The turn acted: its prose is narration in the transcript, not an
            # answer anyone can refuse. Released before the tool cards so it keeps
            # its place above them.
            hold.flush()
        if not tool_calls:
            # No tool call -> the model has produced a final answer.
            answer = msg.get("content", "")
            # Unless it produced nothing at all. A turn with neither prose nor a call
            # is a generation failure, not a conclusion — accepting it ends the run on
            # an empty answer, which is what a hand-off (plan approval → agent mode)
            # looks like when the model returns a single stray token. Drop the empty
            # turn from history so the retry does not build on it.
            if not (answer or "").strip():
                empty_turns += 1
                if hold:
                    hold.discard()
                # Unconditionally, before any branch below decides what happens next.
                # This used to sit inside the retry arm only, so every turn that
                # exhausted the retry budget — and every one the nudge and handback
                # paths below sent round again — left its empty message behind. One
                # session ended up carrying nine of them, showing the model, over and
                # over, that an empty message is an acceptable answer to a reminder.
                if messages and messages[-1].get("role") == "assistant":
                    messages.pop()
                _note_empty_turn(msg, messages, execution_context, step)
                if empty_turns <= AGENT_EMPTY_TURN_RETRIES:
                    emit({"type": "status", "text": (
                        f"  ↻ Empty turn from the model — retrying "
                        f"({empty_turns}/{AGENT_EMPTY_TURN_RETRIES})."
                    )})
                    # The attempt number, so three retries are three different asks
                    # rather than the same sentence sent three times.
                    inject_reminder(messages,
                                    empty_turn_retry_message(execution_context,
                                                             attempt=empty_turns),
                                    category="empty_turn",
                                    tagged=False, execution_context=execution_context,
                                    step=step)
                    step += 1
                    continue
                # Budget spent: end the turn here. Falling through ran the model again
                # through the no-op nudge and the evidence handback below, each of
                # which injects and `continue`s — so a model that had stopped
                # producing anything was asked several more times, leaving one more
                # empty turn behind each time.
                return await _finalize_answer(
                    agent, query,
                    "The model returned empty turns repeatedly and the run was stopped. "
                    "Nothing was concluded — retry the query.",
                    execution_context, messages, logger,
                )
            # A nudge is only worth sending while the model still *acts* on our
            # reminders. Count consecutive bare "done" turns (any tool dispatch
            # below resets it): the first earns the useful reminder, but once the
            # model answers a nudge with another no-op turn — without acting on it —
            # it is ignoring us, so stop re-nudging to avoid echoing the same
            # summary until the caps drain.
            noop_turns = execution_context.get("consecutive_noop_turns", 0) + 1
            execution_context["consecutive_noop_turns"] = noop_turns
            # The mandatory check happens here rather than after each write: the model
            # has stopped editing, so every dirty file is at the revision it will ship
            # at, and a file it went back and forth over is read once instead of once
            # per edit. The stamp inside makes the several gate sites cost one pass.
            sweep_builtin_checks(execution_context)
            if noop_turns <= NUDGE_MAX_CONSECUTIVE_NOOP and maybe_append_nudge(
                agent=agent,
                query=query,
                active_mode=active_mode,
                execution_context=execution_context,
                messages=messages,
            ):
                if hold:
                    hold.discard()
                step += 1
                continue
            if needs_incomplete_finalization(execution_context):
                # Once per query, before the report is assembled: the model has never
                # seen the ledger its summary is about to contradict.
                if not execution_context.get("evidence_handback_used"):
                    execution_context["evidence_handback_used"] = True
                    inject_reminder(
                        messages,
                        evidence_handback_message(execution_context),
                        category="evidence_handback",
                        tagged=False,
                        execution_context=execution_context,
                        step=step,
                    )
                    if hold:
                        hold.discard()
                    step += 1
                    continue
                answer = finalize_incomplete_answer(answer, execution_context)
            answer = await _finalize_answer(agent, query, answer, execution_context, messages, logger)

            # Now prompt for approval if needed
            if agent.approvals and hasattr(agent.approvals, "_pending_review") and agent.approvals._pending_review:
                approved = agent.approvals.flush_pending_review()
                confirm_msg = "Changes applied." if approved else "Changes reverted."
                messages.append({
                    "role": "assistant",
                    "content": confirm_msg
                })
                return answer + "\n" + confirm_msg
            return answer

        # The model acted this turn — reset the no-op streak so it keeps earning
        # reminders as long as it keeps making progress between them.
        execution_context["consecutive_noop_turns"] = 0
        empty_turns = 0
        # Defence in depth for the read-only modes: the write/exec tools were never
        # advertised, but a model can still hallucinate a call to one, and the
        # dual-use exec tool is deliberately still visible for discovery.
        if readonly:
            tool_calls = filter_readonly_tool_calls(
                tool_calls, agent=agent, messages=messages, mode_label=active_mode,
            )
        await _dispatch_tool_calls(tool_calls, agent, messages, execution_context)
        await _post_dispatch_inject(
            agent, messages, execution_context, active_mode=active_mode,
        )
        # Trim/compact again after appending tool results so history stays bounded
        # between iterations. The pre-call enforcement at the top of the loop is
        # what actually guarantees the next prompt fits the window; this keeps the
        # working set small in the meantime. Token counting runs in the worker
        # thread, so a blocking /tokenize round-trip is fine; results are cached
        # per message content, so each tool result is tokenized only once.
        _enforce_context_budget(
            messages, system_content, query_tools, execution_context,
            agent.model, context_mode, compact_fn, _tok,
        )
        step += 1

        # Soft-budget checkpoint: the interactive budget is spent but the hard
        # ceiling is not yet reached — ask the user whether to keep going. A
        # "yes" grants another extension block; a "no" ends the run gracefully.
        if interactive and step >= budget and budget < hard:
            summary = _checkpoint_summary(messages, execution_context, step)
            if bool(agent._request_continue(summary)):
                budget = min(budget + AGENT_STEP_EXTENSION, hard)
                emit({"type": "status", "text": f"  ▸ Continuing — extended to {budget} steps."})
            else:
                emit({"type": "status", "text": "  ▸ Stopping at user request."})
                termination = TERMINATION_USER_STOPPED
                break

    answer = (
        "Stopped at the step checkpoint, at your request."
        if termination == TERMINATION_USER_STOPPED
        else "Reached the maximum number of steps without a final answer."
    )
    sweep_builtin_checks(execution_context)
    if needs_incomplete_finalization(execution_context):
        # A run that ran out of steps is unfinished whatever else the ledger says —
        # in particular it must never borrow the "complete except for what you
        # refused" headline just because a refusal was the only thing recorded.
        answer = finalize_incomplete_answer(answer, execution_context, termination)
    agent.approvals.flush_pending_review()
    answer = await _finalize_answer(agent, query, answer, execution_context, messages, logger)
    return answer


async def run_agent_query(
    *,
    agent: Any,
    query: str,
    max_steps: int = MAX_AGENT_STEPS,
    history: list[dict] | None = None,
    mode: str | None = None,
    thinking: bool = False,
    streaming: bool = True,
    logger: Any = None,
    token_callback: Any = None,
    think_token_callback: Any = None,
    think_start_callback: Any = None,
    think_end_callback: Any = None,
) -> str:
    """Run one user query through plan/agent modes with policy guardrails.

    Builds the per-query execution context and system prompt, then dispatches to the
    plan-mode or agent-mode loop. The two loops (and every exit path) share
    ``_finalize_answer`` for end-of-query bookkeeping.
    """
    agent._tool_cache = {}
    execution_context = agent._new_execution_context()
    agent._apply_carry_context(execution_context)
    validate_execution_context(execution_context)
    active_mode = agent._normalize_mode(mode or agent.mode)
    # Baseline for the live-mode check: only a *later* change to agent.mode is a
    # user switching mid-run. Seeded from the agent, not from active_mode, so an
    # explicit per-query override never registers as one. See _live_mode.
    execution_context[_OBSERVED_MODE] = getattr(agent, "mode", active_mode)

    system_content = await agent._build_system_content(active_mode=active_mode)

    # Store the active todo file path so _sync_checklist can watch it for changes.
    execution_context["todo_file_path"] = agent._get_todo_file()

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
    if not (
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

    # Every loop below mutates this exact list in place, so handing the reference out
    # is enough for a front-end to read the in-flight context without polling the agent.
    agent._live_messages = messages

    # -------------------------------------------------------
    # Skill detection (explicit first, then implicit)
    # -------------------------------------------------------
    skill_name = None

    # Explicit: /skill-name
    if query.strip().startswith("/"):
        candidate = query.strip()[1:].split()[0]
        if getattr(agent, "skills", None) and candidate in agent.skills:
            skill_name = candidate
    else:
        # Implicit detection — pass history so multi-turn context is available
        if getattr(agent, "detect_skill_implicit", None):
            skill_name = await agent.detect_skill_implicit(query, history=history)

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
        execution_context["_skill_suffix"] = system_content[len(_base_system_content):]

    # messages[0] now carries the checklist as it stands at query start; _sync_checklist
    # rewrites it from here on, and only when the file behind it actually changes. The
    # mtime is seeded rather than left unset so an unchanged checklist costs nothing on
    # the first step of every query.
    _todo_fp = execution_context.get("todo_file_path", "")
    if _todo_fp:
        try:
            execution_context["_checklist_stamp"] = os.stat(_todo_fp).st_mtime_ns
        except OSError:
            pass

    # Streaming callbacks bundled once and forwarded to _stream_chat by the loops.
    cb = {
        "token_callback": token_callback,
        "think_token_callback": think_token_callback,
        "think_start_callback": think_start_callback,
        "think_end_callback": think_end_callback,
    }

    try:
        if active_mode == "plan":
            return await _run_plan_mode(
                agent=agent,
                query=query,
                messages=messages,
                execution_context=execution_context,
                max_steps=max_steps,
                thinking=thinking,
                streaming=streaming,
                logger=logger,
                cb=cb,
            )
        return await _run_agent_loop(
            agent=agent,
            query=query,
            active_mode=active_mode,
            messages=messages,
            system_content=system_content,
            execution_context=execution_context,
            max_steps=max_steps,
            thinking=thinking,
            streaming=streaming,
            logger=logger,
            cb=cb,
        )
    finally:
        agent._live_messages = None

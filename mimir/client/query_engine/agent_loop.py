from __future__ import annotations

import os
from typing import Any

from . import streaming as _streaming
from ..event_sink import emit
from .toollist import domains_signaled_by_text, tools_for_context, tools_for_readonly_mode
from .streaming import _process_response, _stream_chat
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
    DOMAIN_REARM_MAX_PER_QUERY,
    NUDGE_MAX_CONSECUTIVE_NOOP,
    THINKING_DEPTH_AUTO,
    max_tools_for,
)
from ..config.models import resolve_pin_role, READONLY_MODES, VALID_MODES
from ..context import validate_execution_context
from ..guardrails.workflow import finalize_incomplete_answer, STEP_LIMIT_NUDGE
from ..prompt.system_prompt import build_discovery_pin_block, _PIN_MARKER
from ..guardrails.nudges import maybe_append_nudge, needs_incomplete_finalization
from ..guardrails.verdict import record_verdict


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


async def _sync_thinking_directive(
    agent: Any,
    messages: list[dict],
    active_mode: str,
    auto_active: bool,
    system_content: str,
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
    build = getattr(agent, "_build_system_content", None)
    if callable(build) and messages and messages[0].get("role") == "system":
        system_content = await build(active_mode=active_mode)
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
        max_tools=max_tools_for(agent.model),
    )


async def _apply_mode_switch(
    agent: Any, messages: list[dict], *, new_mode: str,
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
    system_content = await agent._build_system_content(active_mode=new_mode)
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


def _inject_pin(messages: list[dict], execution_context: dict, pin_role: str = "system"):
    """Append the discovery pin as a TRANSIENT tail message before a model call.

    Keeping the pin at the very end — instead of rewriting the static system
    message (messages[0]) every step — leaves the whole conversation prefix
    byte-stable across the steps of a query, so vLLM automatic prefix caching is
    not invalidated each step. The returned token is handed to :func:`_remove_pin`
    to strip the pin immediately after the call, so it never enters persisted
    history (``_last_full_messages``) or the next step's prefix.

    ``pin_role`` controls attachment (role-sensitivity matters for chat templates):

    * ``"system"`` (default) — a transient system message at the tail. This mirrors
      the existing mid-conversation skill-context system message, so any backend
      that tolerates that tolerates this.
    * ``"user"`` — a transient user message at the tail.
    * ``"append_user"`` — folded onto the end of the last user message's content,
      for strict templates (e.g. Mistral/Devstral) that reject mid-conversation or
      consecutive roles. Falls back to a tail message when there is no user message.

    Returns an opaque token (or ``None`` when the pin is empty).
    """
    pin = build_discovery_pin_block(execution_context)
    if not pin or not pin.strip():
        return None
    if pin_role == "append_user":
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                original = messages[i].get("content", "")
                messages[i]["content"] = (original or "") + "\n" + pin
                return ("append", i, original)
    role = "user" if pin_role == "user" else "system"
    messages.append({"role": role, "content": pin})
    return ("tail", len(messages) - 1, None)


def _remove_pin(messages: list[dict], token) -> None:
    """Undo :func:`_inject_pin` so the transient pin never persists into history."""
    if token is None:
        return
    kind, idx, original = token
    if kind == "tail":
        if messages and str(messages[-1].get("content", "")).startswith(_PIN_MARKER):
            messages.pop()
    elif kind == "append" and 0 <= idx < len(messages):
        messages[idx]["content"] = original





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
    injected messages persist in history (unlike the transient discovery pin). Adjacent
    user turns (e.g. following a post-dispatch nudge) are reconciled downstream by the
    backend's consecutive-user-message merge, so no folding is needed here.
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


def _step_output_text(messages: list[dict], since: int, answer: str) -> str:
    """Concatenate the text this step produced: the model's prose + tool results."""
    parts = [answer or ""]
    for message in messages[since:]:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(p for p in parts if p)


def _maybe_rearm_domains(
    agent: Any,
    query: str,
    execution_context: dict,
    query_tools: list[dict],
    step_text: str,
    active_mode: str = "agent",
) -> list[dict]:
    """Unlock a domain tool group the query never signaled but the work now needs.

    The tool list is frozen per query so the prompt prefix stays cacheable, which means
    a domain pruned at step 0 is invisible for the rest of the run — however clearly the
    work later calls for it. This reopens that door a bounded number of times
    (``DOMAIN_REARM_MAX_PER_QUERY``), driven by what the run itself produced rather than
    by the user's opening wording.

    Returns the tool list to use from here on: the rebuilt one when a domain was
    unlocked, otherwise *query_tools* unchanged (the common case, no cache break).
    """
    if DOMAIN_REARM_MAX_PER_QUERY <= 0:
        return query_tools
    rearmed = execution_context.setdefault("rearmed_domains", set())
    if len(rearmed) >= DOMAIN_REARM_MAX_PER_QUERY:
        return query_tools
    signaled = domains_signaled_by_text(query, step_text, rearmed)
    if not signaled:
        return query_tools

    # Deterministic pick when several domains surface at once: honour the budget by
    # unlocking one per step, in DOMAIN_TOOL_GROUPS order via the sorted key.
    unlocked = sorted(signaled)[: DOMAIN_REARM_MAX_PER_QUERY - len(rearmed)]
    rearmed.update(unlocked)
    # Re-arming reopens a *domain*, never the write/exec surface: going through
    # _mode_tools keeps a read-only mode's filter applied, so the rebuild cannot
    # quietly hand write tools back to the model.
    rebuilt = _mode_tools(agent, query, execution_context, active_mode)
    # The cache break is the whole cost of this feature, so it is reported, not hidden.
    # (This module signals through `emit` rather than a logger, like its siblings; the
    # rebuilt prune set is logged inside toollist.inactive_domain_prefixes.)
    emit({
        "type": "tools_rearmed",
        "domains": unlocked,
        "tool_count": len(rebuilt),
    })
    return rebuilt









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
    # Whether the system message currently carries the "auto" calibration directive.
    # Tracked so a mid-run rung change can rewrite it (see _sync_thinking_directive).
    auto_active = getattr(agent, "thinking_depth", None) == THINKING_DEPTH_AUTO

    # Loop-invariant setup, lifted out of the per-step body.
    _tok = lambda text: _streaming.get_backend().count_text_tokens(agent.model, text)  # noqa: E731
    compact_fn = getattr(agent, "compact_messages", None)
    context_mode = getattr(agent, "context_mode", "full")
    pin_role = resolve_pin_role(agent.model, getattr(agent, "pin_role", ""))
    # Compute the per-query tool list ONCE and reuse it every step. The list is
    # query-stable (pruning + relevance cap depend only on the query, not on the
    # evolving execution_context), so recomputing per step only churned the prompt
    # prefix and broke vLLM prefix caching. The former discover-gated *hiding* of
    # write/external-fetch tools now lives in the call-time policy
    # (check_write_policy / _check_external_fetch), keeping this list stable.
    # A read-only mode (ask) runs this same loop, so its write/exec tools are
    # stripped up front — before the context filter, which only prunes by relevance.
    readonly = active_mode in READONLY_MODES
    query_tools = _mode_tools(agent, query, execution_context, active_mode)

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
            system_content = await _apply_mode_switch(agent, messages, new_mode=active_mode)
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
            messages.append({"role": "user", "content": STEP_LIMIT_NUDGE})

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
        )

        # Enforce the context budget BEFORE every LLM call (not only after tool
        # dispatch) so the first iteration — and any call whose history grew via
        # injected nudges — can never overflow the model window. Accounts for the
        # (stable) tools schema sent alongside `messages`. Runs while the transient
        # discovery pin is NOT in `messages`, so trimming sees the real history.
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

        # Inject the discovery pin as a transient tail message for THIS call only,
        # then strip it immediately (even on cancellation) so it never persists
        # into history or the next step's cacheable prefix.
        pin_token = _inject_pin(messages, execution_context, pin_role)
        try:
            msg = _stream_chat(
                agent.model,
                messages,
                query_tools,
                thinking,
                streaming,
                step_options,
                cancel_flag=getattr(agent, "_cancel_flag", None),
                **cb,
            )
        finally:
            _remove_pin(messages, pin_token)
        _process_response(msg, messages, thinking, streamed_thinking=(streaming and cb["think_token_callback"] is not None))

        # Before the turn is judged terminal, so a verdict stated only in the final
        # answer still counts. Reads `content`, never the thinking block: a judgement
        # kept in private reasoning is not a statement, and the user never sees it.
        record_verdict(msg.get("content", ""), execution_context)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            # No tool call -> the model has produced a final answer.
            answer = msg.get("content", "")
            # A nudge is only worth sending while the model still *acts* on our
            # reminders. Count consecutive bare "done" turns (any tool dispatch
            # below resets it): the first earns the useful reminder, but once the
            # model answers a nudge with another no-op turn — without acting on it —
            # it is ignoring us, so stop re-nudging to avoid echoing the same
            # summary until the caps drain.
            noop_turns = execution_context.get("consecutive_noop_turns", 0) + 1
            execution_context["consecutive_noop_turns"] = noop_turns
            if noop_turns <= NUDGE_MAX_CONSECUTIVE_NOOP and maybe_append_nudge(
                agent=agent,
                query=query,
                active_mode=active_mode,
                execution_context=execution_context,
                messages=messages,
            ):
                step += 1
                continue
            if needs_incomplete_finalization(execution_context):
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
        # Defence in depth for the read-only modes: the write/exec tools were never
        # advertised, but a model can still hallucinate a call to one, and the
        # dual-use exec tool is deliberately still visible for discovery.
        if readonly:
            tool_calls = filter_readonly_tool_calls(
                tool_calls, agent=agent, messages=messages, mode_label=active_mode,
            )
        _pre_dispatch_len = len(messages)
        await _dispatch_tool_calls(tool_calls, agent, messages, execution_context)
        await _post_dispatch_inject(agent, messages, execution_context)
        # What the step produced can reveal a tool domain the opening query never
        # mentioned; re-arming it here is the one sanctioned break of the frozen list.
        query_tools = _maybe_rearm_domains(
            agent, query, execution_context, query_tools,
            _step_output_text(messages, _pre_dispatch_len, msg.get("content", "")),
            active_mode=active_mode,
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
                break

    answer = "Reached the maximum number of steps without a final answer."
    if needs_incomplete_finalization(execution_context):
        # A run that ran out of steps is unfinished whatever else the ledger says —
        # in particular it must never borrow the "complete except for what you
        # refused" headline just because a refusal was the only thing recorded.
        answer = finalize_incomplete_answer(answer, execution_context, ran_out_of_steps=True)
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
    # Lazily build the structural baseline on the first repo-touching query (no-op for
    # pure-math/bibliography queries, and idempotent after the first build). Must run
    # before its consumers: the execution-context seed and the foundational repo-structure
    # section injected by _build_system_content.
    await agent._ensure_repo_baseline(query)
    agent._seed_execution_context_from_baseline(execution_context)
    agent._apply_carry_context(execution_context)
    validate_execution_context(execution_context)
    active_mode = agent._normalize_mode(mode or agent.mode)
    # Baseline for the live-mode check: only a *later* change to agent.mode is a
    # user switching mid-run. Seeded from the agent, not from active_mode, so an
    # explicit per-query override never registers as one. See _live_mode.
    execution_context[_OBSERVED_MODE] = getattr(agent, "mode", active_mode)

    system_content = await agent._build_system_content(active_mode=active_mode)

    # Store the active todo file path so the per-step pin can show live progress.
    execution_context["todo_file_path"] = agent._get_todo_file()

    # The caller may or may not have already appended the current user turn to
    # ``history``: the WS server does (it owns the chat history and appends the
    # message before dispatching), the CLI does not. Only add the query ourselves
    # when history doesn't already end with this exact user message — otherwise the
    # prompt carries two consecutive identical ``user`` turns. Tolerant templates
    # (Ollama/Qwen) ignore that, but the Mistral tokenizer (Devstral in native
    # mistral mode) treats the illegal role sequence as garbage and degenerates
    # into token salad.
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
        messages.append({
                "role": "system",
                "content": (
                    "SKILL CONTEXT (SUBORDINATE). "
                    "The base system instructions remain fully authoritative. "
                    "Apply this methodology where relevant. "
                    + skill["content"]
                ),
            })

    # The discovery pin (carry-context knowledge: existing paths, prior reads) is
    # injected as a transient tail message on every step including the first (see
    # _inject_pin), so the static system message at messages[0] stays byte-stable
    # across the whole query for prefix caching.

    # Streaming callbacks bundled once and forwarded to _stream_chat by the loops.
    cb = {
        "token_callback": token_callback,
        "think_token_callback": think_token_callback,
        "think_start_callback": think_start_callback,
        "think_end_callback": think_end_callback,
    }

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

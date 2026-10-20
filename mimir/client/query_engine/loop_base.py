"""What every step loop shares: the turn's state and one model call per step.

A :class:`StepLoop` owns what a step does whatever the mode; a subclass decides what its
mode does with the answer. A mode change returns a :class:`ModeSwitch` to the driver
(``turn.run_loops``), so loops never call one another.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from . import streaming as _streaming
from ..config.limits import AGENT_EMPTY_TURN_RETRIES
from ..config.models import READONLY_MODES, VALID_MODES
from ..config.settings import THINKING_DEPTH_AUTO
from ..protocol.event_sink import emit
from ..guardrails.nudges import drop_transient_reminders, inject_reminder
from ..guardrails.nudges.messages import empty_turn_retry_message
from .deferral import end_deferred_calls, end_deferred_turn
from .dispatch import dispatch_tool_calls, post_dispatch_inject
from .finalize import finalize_answer
from .history import enforce_context_budget
from .readonly_guard import filter_readonly_tool_calls
from .streaming import DraftHold, note_truncated_turn, process_response, stream_chat
from ..context.execution_context import ExecutionContext
from ..core.ports import AgentPort


@dataclass
class TurnContext:
    """One query, as every loop that serves it sees it.

    Built once by ``turn.run_agent_query`` and handed from loop to loop when the mode
    changes, so what the turn has learned — its messages, its execution context, the
    state of its system message — survives the handoff.
    """

    agent: AgentPort
    query: str
    messages: list[dict]
    execution_context: ExecutionContext
    # 0 means no ceiling: the run ends when the model answers, a guard stops it, or the
    # user interrupts. Callers that want a bound (runner, sub-agents, tests) pass one.
    max_steps: int = 0
    thinking: bool = False
    streaming: bool = True
    logger: Any = None
    # token_callback / think_token_callback / think_start_callback / think_end_callback.
    callbacks: dict = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.agent.model

    def count_tokens(self, text: str) -> int:
        return _streaming.get_backend().count_text_tokens(self.agent.model, text)


@dataclass
class ModeSwitch:
    """A loop handing the turn over to the loop of *mode*, with its system message."""

    mode: str
    system_content: str


# ── Live settings ───────────────────────────────────────────────────────────────
# The user can change these while a query is already running (the WS command handler
# and the CLI mutate the agent from outside the loop), and the change has to land on
# the next step rather than the next query. A setting never chosen (None) keeps the value
# the caller passed in.

def _live_thinking(agent: AgentPort, fallback: bool) -> bool:
    """The thinking flag to use for the *next* model call."""
    val = agent.thinking
    return fallback if val is None else bool(val)


def _live_thinking_budget(agent: AgentPort) -> int:
    """The token budget for the next model call. ``-1`` = unbudgeted."""
    val = agent.thinking_budget
    return val if isinstance(val, int) else -1


def _live_temperature(agent: AgentPort) -> float | None:
    """The temperature the user set for this model, re-read per call; None = its own."""
    val = agent.temperature
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    return float(val)


def _live_mode(agent: AgentPort, active_mode: str, execution_context: ExecutionContext) -> str:
    """The mode for the next step: *active_mode*, unless ``agent.mode`` changed since last seen.

    A context never seeded takes the agent's mode as its baseline.
    """
    val = agent.mode.strip().lower()
    if execution_context.bookkeeping.observed_agent_mode is None:
        execution_context.bookkeeping.observed_agent_mode = val
        return active_mode
    if val not in VALID_MODES or val == execution_context.bookkeeping.observed_agent_mode:
        return active_mode
    execution_context.bookkeeping.observed_agent_mode = val
    return val


# ── The system message ──────────────────────────────────────────────────────────

async def rebuild_system_content(
    agent: AgentPort, active_mode: str, execution_context: ExecutionContext,
) -> str:
    """The system message after any rebuild: the prompt plus the loaded skill's block.

    Every rebuild site goes through here, so none drops the skill.
    """
    content = await agent.build_system_content(active_mode=active_mode)
    return content + (execution_context.bookkeeping.skill_suffix or "")


async def _sync_thinking_directive(
    agent: AgentPort,
    messages: list[dict],
    active_mode: str,
    auto_active: bool,
    system_content: str,
    execution_context: ExecutionContext,
) -> tuple[bool, str]:
    """Rewrite the system message when the run enters or leaves the "auto" rung.

    That rung's calibration directive lives in the prompt; returns ``(auto, system_content)``.
    """
    depth = agent.thinking_depth
    if depth is None:
        return auto_active, system_content
    now = depth == THINKING_DEPTH_AUTO
    if now == auto_active:
        return auto_active, system_content
    if messages and messages[0].get("role") == "system":
        system_content = await rebuild_system_content(
            agent, active_mode, execution_context,
        )
        messages[0]["content"] = system_content
    return now, system_content


async def _apply_mode_switch(
    agent: AgentPort, messages: list[dict], *, new_mode: str, execution_context: ExecutionContext,
) -> str:
    """Rebuild the mode-dependent system prompt after a mid-run mode change; return it."""
    emit({"type": "status", "text": f"  ↻ Switched to {new_mode} mode mid-run"})
    emit({"type": "mode", "mode": new_mode})
    system_content = await rebuild_system_content(agent, new_mode, execution_context)
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = system_content
    return system_content



# ── Steering and diagnostics ────────────────────────────────────────────────────

def drain_steer(agent: AgentPort, messages: list[dict]) -> None:
    """Append the messages the user typed mid-run as user turns, at a step boundary.

    Through the optional ``poll_steer`` hook. Nothing may follow a steer before the call:
    the last position is what the model answers.
    """
    poll = agent.poll_steer
    if not poll:
        return
    for text in (poll() or []):
        text = (text or "").strip()
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        emit({"type": "steer_injected", "text": text})


def _note_empty_turn(msg: dict, messages: list[dict], execution_context: ExecutionContext,
                     step: int) -> None:
    """Record the shape of the prompt that produced an empty turn, reminder categories included."""
    tail = " → ".join(str(m.get("role")) for m in messages[-6:])
    # Read after the fact: the reminders that were in this prompt have already been
    # taken back out by the time the turn is judged empty.
    cats = list(execution_context.nudges.last_call_reminders or [])
    emit({"type": "status", "text": (
        f"  ⓘ Empty turn diagnostics — step {step}, {len(messages)} messages, "
        f"finish_reason={msg.get('finish_reason') or 'none'}, "
        f"reminders in prompt={len(cats)}"
        + (f" ({', '.join(sorted(set(cats)))})" if cats else "")
        + f", tail: {tail}"
    )})


# ── The loop ────────────────────────────────────────────────────────────────────

class StepLoop:
    """One mode's step loop over a :class:`TurnContext`.

    ``run`` returns the answer (a ``str``) or a :class:`ModeSwitch`. Subclasses
    implement :meth:`run`, built from the steps below.
    """

    # What a run that gives up on empty turns says it did not do.
    EMPTY_TURNS_ANSWER = ("The model returned empty turns repeatedly and the run was "
                          "stopped. Nothing was concluded — retry the query.")

    def __init__(self, turn: TurnContext, mode: str, system_content: str) -> None:
        self.turn = turn
        self.agent = turn.agent
        self.messages = turn.messages
        self.ec = turn.execution_context
        self.mode = mode
        self.system_content = system_content
        self.thinking = turn.thinking
        self.step = 0
        self.empty_turns = 0
        # Whether the system message carries the "auto" calibration directive, so a
        # mid-run rung change can rewrite it (see _sync_thinking_directive).
        self.auto_active = self.agent.thinking_depth == THINKING_DEPTH_AUTO
        # Read once: properties of the run. Re-read per step, a mid-run change would
        # move the budget the history was already trimmed against.
        self.context_mode = self.agent.context_mode
        self.compact_fn = self.agent.compact_messages
        # An agent may be held below the model's window (a sub-agent is: see
        # server_spawn_agent.SUBAGENT_CONTEXT_TOKENS_*).
        self.context_ceiling = int(self.agent.max_context_tokens or 0) or None
        # No sampling params by default: the server applies the model's own
        # generation_config. A sub-agent caps its own answer: left to the backend
        # default it gets the whole answer reserve, and a step that runs away is
        # invisible from outside — one mute row for as long as it takes to generate.
        self.base_options: dict = {}
        answer_cap = int(self.agent.max_answer_tokens or 0)
        if answer_cap > 0:
            self.base_options["max_tokens"] = answer_cap

    @property
    def readonly(self) -> bool:
        return self.mode in READONLY_MODES

    @property
    def unbounded(self) -> bool:
        return self.turn.max_steps <= 0

    def steps_left(self) -> bool:
        return self.unbounded or self.step < self.turn.max_steps

    async def run(self) -> str | ModeSwitch:
        raise NotImplementedError

    # ── step pieces ─────────────────────────────────────────────────────────────

    def serves(self, mode: str) -> bool:
        """Whether this loop can run *mode* itself, without handing the turn over."""
        return mode == self.mode

    async def follow_mode(self) -> ModeSwitch | None:
        """Pick up a mode the user switched to since the last step.

        The system message is rebuilt for the new mode either way; a mode this loop
        serves is taken up in place (:meth:`on_mode_changed`), any other hands the turn
        over.
        """
        live = _live_mode(self.agent, self.mode, self.ec)
        if live == self.mode:
            return None
        self.system_content = await _apply_mode_switch(
            self.agent, self.messages, new_mode=live, execution_context=self.ec,
        )
        if not self.serves(live):
            return ModeSwitch(live, self.system_content)
        self.mode = live
        self.on_mode_changed()
        return None

    def on_mode_changed(self) -> None:
        """React to a mode this loop serves being switched to. Default: nothing."""

    async def refresh_live_settings(self) -> None:
        """Re-read the reasoning depth so a rung moved mid-run lands on this call."""
        self.thinking = _live_thinking(self.agent, self.thinking)
        self.auto_active, self.system_content = await _sync_thinking_directive(
            self.agent, self.messages, self.mode, self.auto_active,
            self.system_content, self.ec,
        )

    def thinking_budget(self, base: int) -> int:
        """The thinking budget for this step, from the rung's. Default: unchanged."""
        return base

    def step_options(self) -> dict:
        options = dict(self.base_options)
        budget = _live_thinking_budget(self.agent)
        if self.thinking and budget > 0:
            options["thinking_budget"] = self.thinking_budget(budget)
        temperature = _live_temperature(self.agent)
        if temperature is not None:
            options["temperature"] = temperature
        return options

    def enforce_budget(self, tools: list[dict]) -> None:
        """Fit the history to the window the next call will be measured against."""
        enforce_context_budget(
            self.messages, self.system_content, tools, self.ec,
            self.turn.model, self.context_mode, self.compact_fn, self.turn.count_tokens,
            context_ceiling=self.context_ceiling,
        )

    async def call_model(self, tools: list[dict], *, may_refuse: bool
                         ) -> tuple[dict, DraftHold | None]:
        """One model call, within the budget; returns the message and the hold its prose sits in.

        When the loop *may_refuse* a bare answer, its prose is held off the screen.
        """
        self.enforce_budget(tools)
        callbacks = self.turn.callbacks
        hold = (DraftHold(callbacks["token_callback"])
                if may_refuse and callbacks.get("token_callback") is not None else None)
        step_callbacks = {**callbacks, "token_callback": hold.capture} if hold else callbacks
        try:
            msg = stream_chat(
                self.turn.model, self.messages, tools, self.thinking,
                self.turn.streaming, self.step_options(),
                cancel_flag=self.agent.turn.cancel_flag,
                on_retry=hold.discard if hold else None,
                **step_callbacks,
            )
        except BaseException:
            # Cancelled, or the backend gave up: no guardrail will get to refuse this
            # turn, so whatever prose was held is the user's to keep.
            if hold:
                hold.flush()
            raise
        finally:
            # The reminders injected for THIS call have now been put to the model.
            # Keeping them is what let one sentence reach 21 identical copies in a
            # single session; the emitted nudge_injected events keep the diagnosis.
            drop_transient_reminders(self.messages, self.ec)
        process_response(msg, self.messages, self.thinking, streamed_thinking=(
            self.turn.streaming and callbacks.get("think_token_callback") is not None))
        note_truncated_turn(msg, self.step)
        if hold and msg.get("tool_calls"):
            hold.flush()
        return msg, hold

    async def on_empty_turn(self, msg: dict, hold: DraftHold | None) -> str | None:
        """A turn with neither prose nor a call: drop it, then retry differently or end the run.

        Past the budget the run ends with :attr:`EMPTY_TURNS_ANSWER`.
        """
        self.empty_turns += 1
        if hold:
            hold.discard()
        if self.messages and self.messages[-1].get("role") == "assistant":
            self.messages.pop()
        _note_empty_turn(msg, self.messages, self.ec, self.step)
        if self.empty_turns > AGENT_EMPTY_TURN_RETRIES:
            return await self.finalize(self.EMPTY_TURNS_ANSWER)
        emit({"type": "status", "text": (
            f"  ↻ Empty turn from the model — retrying "
            f"({self.empty_turns}/{AGENT_EMPTY_TURN_RETRIES})."
        )})
        inject_reminder(self.messages,
                        empty_turn_retry_message(self.ec, attempt=self.empty_turns),
                        category="empty_turn", tagged=False,
                        execution_context=self.ec, step=self.step)
        return None

    def prepare_calls(self, tool_calls: list[Any]) -> list[Any]:
        """The calls to run, from the ones the model made. Default: the read-only modes
        drop what they may not run (defence in depth: write/exec tools are never
        advertised there, but a model can still hallucinate a call to one)."""
        if self.readonly:
            return filter_readonly_tool_calls(
                tool_calls, agent=self.agent, messages=self.messages,
                mode_label=self.mode,
            )
        return tool_calls

    async def dispatch(self, tool_calls: list[Any]) -> str | None:
        """Run the step's calls. Returns an answer when the turn ends here.

        It does when a call waited on a user who has left: the step stops, and the
        user's answer resumes it (see deferral).
        """
        self.empty_turns = 0
        tool_calls = self.prepare_calls(tool_calls)
        await dispatch_tool_calls(tool_calls, self.agent, self.messages, self.ec)
        if end_deferred_calls(self.agent, self.messages, tool_calls,
                              query=self.turn.query, mode=self.mode):
            return end_deferred_turn(self.agent, self.ec)
        # The mid-loop correctives: an alert the dispatch staged (a repeated failing
        # call) reaches the model here, at the step it concerns.
        await post_dispatch_inject(self.agent, self.messages, self.ec,
                                    active_mode=self.mode)
        return None

    async def finalize(self, answer: str) -> str:
        return await finalize_answer(self.agent, self.turn.query, answer, self.ec,
                                      self.messages, self.turn.logger)

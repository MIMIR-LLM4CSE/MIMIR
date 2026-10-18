"""What every handler of a connection shares: the conversation on screen as a service.

Saving it, listing the conversations, the context bar, a command's reply, the agent of
the conversation on screen. Each handler group is built with this, the view state, the
socket and the pool, and with nothing of the others.
"""
from __future__ import annotations

import logging
from typing import Any

from ....session.conversation import Conversation
from ..pool.pool import AgentPool
from ..runtime import context_budget_for, get_backend
from ..worker.worker import AgentWorker
from .view import Sender, ViewState

logger = logging.getLogger(__name__)

# Events that must reach the user whichever conversation they belong to: each is a
# question an agent is parked on, and withholding it as foreign (which it is, during a
# background-job wake in another session) would leave the turn waiting on an answer the
# user was never shown. ``prompt_expired`` takes one of those cards back off the screen.
INTERACTION_EVENTS = frozenset({"approval", "user_question", "prompt_expired"})


class Screen:
    def __init__(self, view: ViewState, out: Sender, pool: AgentPool) -> None:
        self.view = view
        self.out = out
        self.pool = pool

    @property
    def worker(self) -> AgentWorker:
        """The agent of the conversation on screen, or the pool's detached stand-in.

        One worker per conversation, built on its first query. Until then the stand-in
        answers every getter with the pre-agent value and ignores every setter; a query
        and a setting that must outlive it go through the pool explicitly.
        """
        return self.pool.worker_or_detached(self.view.active_session_id)

    # ── The conversation on screen ───────────────────────────────────────────

    @property
    def conv(self) -> Conversation:
        """The conversation on screen. Set by the handshake, before any message is read."""
        if self.view.conv is None:
            raise RuntimeError("no conversation on screen yet")
        return self.view.conv

    @property
    def session_id(self) -> str:
        """The id of the conversation on screen (see :attr:`conv`)."""
        return self.conv.id

    def save(self) -> None:
        """Write the conversation on screen down, with how far this client has rendered
        it — never backwards: a replay is sized from it, and lowering it would resend
        what the client already has."""
        conv = self.view.conv
        if conv is None:
            return
        record = conv.record
        record.rendered_seq = max(record.rendered_seq, self.view.rendered_seq)
        self.pool.writer.save_conversation(conv)

    def running_turn_is_ours(self) -> bool:
        """Whether the conversation on screen has a turn in flight. Several can be
        working at once; a message typed here steers this one's turn or starts one."""
        return self.pool.is_busy(self.view.active_session_id)

    def is_foreign_event(self, ev: dict) -> bool:
        """True when *ev* was produced for a session other than the one on screen.

        The stamp keeps one conversation's output out of another's chat; it goes to its
        own transcript instead. Unstamped events and the interaction events always pass.
        """
        if ev.get("type") in INTERACTION_EVENTS:
            return False
        ev_session = ev.get("session_id")
        return ev_session is not None and ev_session != self.view.active_session_id

    def apply_setting(self, name: str, value: Any) -> str:
        """Apply a UI knob to every live conversation and to the next agent built.

        Returns the first rejection a live worker reported, or "".
        """
        return next(iter(self.pool.apply_setting(name, value)), "")

    # ── The conversation list ────────────────────────────────────────────────

    async def send_sessions_list(self) -> None:
        store = self.pool.conversations.store
        sessions = store.list_sessions()
        # The conversation on screen before it is saved, so the panel shows it at once.
        conv = self.view.conv
        if conv is not None and not conv.saved:
            if conv.id not in {s.id for s in sessions}:
                sessions = [conv.meta()] + sessions
        await self.out.try_send({
            "type": "sessions_list",
            "sessions": [{**s.to_dict(), **self.session_activity(s.id)} for s in sessions],
        })

    def session_activity(self, session_id: str) -> dict:
        """What a conversation is doing right now, for its row in the panel.

        ``running`` is a turn in flight, ``parked`` one waiting on a card, ``runs`` the
        background runs it still has going — a conversation that launched a build and
        answered has no turn in flight, and that is the case detaching exists for.
        Guarded: read for every row, and a hiccup must lose one dot, not the panel.
        """
        try:
            worker = self.pool.get(session_id)
            runs = len(worker.watched_job_keys()) if worker is not None else 0
            return {
                "running": self.pool.is_busy(session_id),
                "parked": self.pool.is_parked(session_id),
                "queued": self.pool.queued_position(session_id) is not None,
                "runs": runs,
            }
        except Exception:
            logger.debug("session_activity: a best-effort step failed and was skipped", exc_info=True)
            return {}

    def session_title(self, session_id: str) -> str:
        """A conversation's name, or "" if it has none (or none saved yet)."""
        conv = self.pool.conversations.get(session_id)
        return conv.title if conv is not None else ""

    def publish_title(self, session_id: str | None, title: str) -> None:
        """Tell a conversation's agent what it is called, so a card it raises while the
        user reads elsewhere can say who is asking."""
        worker = self.pool.get(session_id)
        if worker is not None:
            worker.session_title = title or ""

    # ── The context bar ──────────────────────────────────────────────────────

    def ctx_budget(self) -> tuple[int, int]:
        """(total_tokens, reserved_tokens) for the current context mode."""
        conv = self.view.conv
        default = conv.record.context_mode if conv is not None else "full"
        mode = self.worker.setting("context_mode", default)
        total, reserved, _, _ = context_budget_for(self.worker.model, mode)
        return total, reserved

    async def emit_context_usage(self) -> None:
        """Push a context_usage event to the client (best-effort, never raises)."""
        try:
            conv = self.view.conv
            if conv is None:
                return
            total, reserved = self.ctx_budget()
            # While a query runs, the transcript that matters is the agent's in-flight
            # one: the conversation only gains the turn once the answer lands.
            messages = self.worker.live_history()
            if messages is None:
                messages = conv.history
            # allow_network=False: on the event loop, which must not block on tokenize.
            history_used = get_backend().count_messages_tokens(
                self.worker.model, messages, allow_network=False)
            # The fixed per-call overhead (system prompt + tools schema) rides on every
            # call; without it the bar hides the tokens that actually overflow.
            overhead = self.worker.usage.overhead_tokens()
            used = history_used + overhead
            if used == self.view.last_context_usage:
                return
            self.view.last_context_usage = used
            await self.out.send({
                "type": "context_usage",
                "used_tokens": used,
                "total_tokens": total,
                "reserved_tokens": reserved,
                "overhead_tokens": overhead,
                "overhead_measured": self.worker.usage.overhead_is_measured(),
                # No agent yet, so the fixed part is not in `used_tokens` at all.
                "provisional": overhead == 0,
                # What the model has this turn, against the untrimmed record behind it.
                "history_messages": len(conv.history),
                "history_messages_full": len(conv.history_full),
            })
        except Exception:
            logger.debug("emit_context_usage: the client went away mid-send", exc_info=True)

    # ── Replies ──────────────────────────────────────────────────────────────

    async def command_reply(self, command: str, title: str, *,
                            items: list[dict] | None = None, note: str = "",
                            tone: str = "ok", transient: bool = False) -> None:
        """One structured answer for a session command: a listing, or something
        irreversible that just happened (a setting reports its state instead).

        ``tone``: ``ok``, ``warn`` (irreversible), ``empty`` (a listing with nothing in
        it), ``quiet`` (a state the user walked back into, drawn as a rule). ``items``
        are ``{label, detail}`` rows. ``transient`` keeps it out of the stored
        transcript, for a card about this moment rather than the conversation.
        """
        await self.out.send({
            "type": "command_output",
            "command": command,
            "title": title,
            "items": items or [],
            "note": note,
            "tone": tone,
            "transient": transient,
        })

    async def resend_deferred_prompt(self) -> None:
        """Put the card a deferred turn of this conversation waits on back on screen."""
        conv = self.view.conv
        prompt = ((conv.pending_interaction if conv is not None else None) or {}).get("prompt")
        if prompt:
            await self.out.try_send(prompt)

    # ── Agents ───────────────────────────────────────────────────────────────

    async def ensure_worker(self) -> AgentWorker | None:
        """This conversation's agent, built now if it has none and there is room.

        None means the pool is full of conversations that are working, waiting on the
        user or watching a job; the caller queues and says so. The build runs off the
        event loop and is announced, since it is the backend wait plus ~19 servers.
        """
        session_id = self.view.active_session_id
        if not session_id:
            return None
        existing = self.pool.get(session_id)
        if existing is not None:
            return existing

        def _announce() -> None:
            self.out.schedule(self.out.send({
                "type": "status",
                "session_id": session_id,
                "text": "  ⏳ Starting an agent for this conversation…",
            }))

        try:
            worker = await self.pool.worker_for(session_id, on_wait=_announce)
            if worker is not None:
                worker.session_title = self.session_title(session_id)
            return worker
        except Exception as exc:
            await self.out.send({
                "type": "error",
                "session_id": session_id,
                "text": f"Could not start an agent for this conversation: {exc}",
            })
            return None

    async def admit_waiting(self) -> None:
        """Start the turn of a conversation that was waiting for an agent slot.

        Called the moment a turn ends, which is when a slot can free; the idle sweep
        would make the queued conversation wait out its interval for nothing.
        """
        try:
            for session_id in await self.pool.pump():
                await self.out.send({
                    "type": "status",
                    "session_id": session_id,
                    "text": "  ▶ An agent slot freed — this conversation is starting.",
                })
        except Exception:
            logger.warning("could not admit a queued conversation", exc_info=True)

    @staticmethod
    def text_count(messages: list) -> int:
        """Number of plain-text bubbles — the part of a transcript both sides share.

        A note this layer wrote itself (``role: "system"``) never reached the webview,
        so counting it would make every transcript the client sends look short by one.
        """
        return sum(1 for m in messages
                   if isinstance(m, dict) and m.get("kind", "text") == "text"
                   and m.get("role") != "system")

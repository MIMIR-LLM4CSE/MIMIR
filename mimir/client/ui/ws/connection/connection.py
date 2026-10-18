"""One WebSocket connection.

``Connection`` owns one WebSocket connection: which conversation it shows, the drain
loop that forwards the stream to the client, and the inbound-message dispatch table.
The conversation itself is the pool's (``session.conversation``): this view writes
what the user does to it, and renders what the agents did. The wire protocol is
documented in ``server.main``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ....session.journal import read_since
from ..pool.pool import AgentPool
from ..worker.worker import AgentWorker
from .handlers.base import Handlers
from .handlers.cards import CardHandlers
from .handlers.conversations import ConversationHandlers
from .handlers.detach import DetachHandlers
from .handlers.panel import PanelHandlers
from .handlers.turns import TurnHandlers
from .router import EventRouter
from .screen import Screen
from .view import Sender, ViewState

# The websocket connection handle passed in by the server (websockets library type
# varies by version); kept as an untyped alias so the annotation resolves.
_WS = Any

logger = logging.getLogger(__name__)


# How many journal events one ``replay`` frame carries. The frame cap is 32 MiB, which a
# long detached run's transcript can exceed whole; chunking also lets the webview render
# the first of them while the rest are still arriving.
_REPLAY_CHUNK = 500

# The most a replay will resend. Past this the oldest are dropped and the frame says so:
# what a returning user needs is the end of the run, and an unbounded replay into the
# webview's reducer is how a twelve-hour absence wedges the panel.
_REPLAY_MAX_EVENTS = 5000

class Connection:
    """One WebSocket connection: its view state, its handler groups, its stream."""

    def __init__(self, ws: "_WS", pool: "AgentPool") -> None:
        self.pool = pool
        self.store = pool.conversations.store
        self.view = ViewState()
        self.out = Sender(ws)
        self.screen = Screen(self.view, self.out, pool)
        parts = (self.view, self.out, pool, self.screen)
        self.conversations = ConversationHandlers(*parts)
        self.turns = TurnHandlers(*parts)
        self.cards = CardHandlers(*parts)
        self.detach = DetachHandlers(*parts)
        self.panel = PanelHandlers(*parts)
        # Inbound message type -> the group and the method that serve it.
        self._handlers: dict[str, tuple[Handlers, str]] = {
            kind: (group, method)
            for group in (self.conversations, self.turns, self.cards, self.detach, self.panel)
            for kind, method in group.MESSAGES.items()
        }

    @property
    def ws(self) -> "_WS":
        """The socket (``Sender.ws``)."""
        return self.out.ws

    @property
    def worker(self) -> AgentWorker:
        """The agent of the conversation on screen, or None if it has none yet.

        A property, not a field: there is one worker per conversation, built on that
        conversation's first query, and every ``self.worker.*`` call site asks about the
        session being read — which is what this answers.

        Until that first query the conversation has no worker, and this answers with the
        pool's detached stand-in rather than None: every getter on it gives the right
        pre-agent answer and every setter is a no-op. Two things must not be silently
        dropped that way — submitting a query, and a setting that has to survive into the
        next worker built — and both go through the pool explicitly.
        """
        return self.screen.worker

    def _greeting(self) -> dict:
        """The ``ready`` sent the moment the socket is accepted.

        Its ``agent_ready`` is the part worth stating rather than leaving to be
        inferred. This is sent well before the agent exists — the worker is still
        waiting on the LLM backend, minutes of it on a cold vLLM — and the worker
        sends a second greeting, spelled "ready" too, once it really is up. The two
        were indistinguishable to the client, so the chat invited the user to type
        into an agent that could not answer. The worker's own says True; this one
        says what is actually so.
        """
        return {"type": "ready", **self.pool.ui_state(self.view.active_session_id)}

    async def run(self) -> None:
        # Subscribe BEFORE anything else, and this order is the point: from here on
        # nothing a turn produces can be missed, because it accumulates in this
        # subscription's queue while the rest of the handshake runs. Nothing has to be
        # swept either — the pump has already drained and journaled everything, so what
        # a worker holds is never debris.
        self.view.sub = self.pool.bus.subscribe()
        # Named out here, created inside the guarded body below: the ``finally`` has to
        # be able to cancel them whether or not the handshake reached the line that
        # starts them.
        drain_task: asyncio.Task | None = None
        jobs_task: asyncio.Task | None = None
        # Everything from the subscription on is guarded, and that is the point. A
        # subscription is what tells the pool a socket will route this process's output
        # — the committer and the headless wake consumer both stand down while one
        # exists — so a view that ends without closing it leaves the server believing
        # somebody is watching for as long as it lives: no turn committed, no finished
        # run handed to its conversation, no check-in. Every step of the handshake sends
        # on a socket that can close under it (a window shut mid-replay raises here),
        # which makes an unguarded exit the ordinary case rather than the exotic one.
        try:
            # Send ready immediately so the webview transitions out of "connecting".
            greeting = self._greeting()
            try:
                await self.ws.send(json.dumps(greeting))
            except Exception:
                logger.debug("run: the client went away mid-send", exc_info=True)
                return

            # ── Session initialisation ────────────────────────────────────────────
            # Purge any empty sessions left over from previous (pre-fix) reconnects.
            self.conversations.purge_empty_sessions()
            await self.screen.send_sessions_list()
            await self.panel.send_toggles()

            sessions = self.store.list_sessions()
            if sessions:
                # Auto-load the most recent session.
                try:
                    await self.conversations.load_session(sessions[0].id)
                except Exception:
                    await self.conversations.create_new_session()
            else:
                await self.conversations.create_new_session()

            # What happened while nobody was attached, before the cards: a replay under a
            # chat that is already on screen, and a card on top of that.
            await self.send_replay()

            # The run never stopped, so its own level is the truth the panel must show.
            await self.detach.restore_detached_autonomy()

            # Last, so the card lands under a chat that is already on screen.
            await self.cards.resend_parked_prompt()

            await self.panel.send_served_models()

            # The greeting above may have said "not ready" and the worker may have come up
            # since. It announces that once, by queueing a second ``ready``, and an
            # announcement made before this subscription existed is one nothing will repeat:
            # the chat would sit on "starting the agent" for the life of the socket with a
            # working agent behind it. Asking the worker directly costs one message.
            if not greeting["agent_ready"] and self.worker.agent_ready():
                try:
                    await self.ws.send(json.dumps(self._greeting()))
                except Exception:
                    logger.debug("run: the client went away mid-send", exc_info=True)
                    return

            drain_task = asyncio.create_task(self._drain_loop())
            # The runs that ended while nothing was listening, off the critical path and
            # deliberately so. It walks every session's job directories, and a job
            # directory is never swept however old — so on a long-lived workspace this is
            # unbounded file work. Awaited here, inside the handshake, it sat between the
            # client connecting and this loop reading its first message: the chat came up,
            # said it was ready, and the query was never read. A task cannot do that.
            jobs_task = asyncio.create_task(self.detach.report_ended_jobs())
            try:
                async for raw in self.ws:
                    await self.handle(raw)
            except Exception:
                logger.debug("run: a best-effort step failed and was skipped", exc_info=True)
        finally:
            summary_task = self.conversations.summary_task
            if summary_task is not None:
                summary_task.cancel()
            for task in (drain_task, jobs_task):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            # The agents and the pump live on; only this view of them ends here.
            if self.view.sub is not None:
                self.view.sub.close()

    async def _drain_loop(self) -> None:
        """Forward this connection's share of the stream to the client (``EventRouter``).

        ``run()`` subscribes before the handshake so nothing can be missed; the router
        subscribes itself for a caller that starts the loop on its own.
        """
        await EventRouter(self).run()

    async def handle(self, raw: str) -> None:
        """Serve one inbound message. An unknown type is ignored; a failing one is
        reported to the client and the connection reads on."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        served = self._handlers.get(msg.get("type"))
        if served is None:
            return
        group, method = served
        try:
            await getattr(group, method)(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # One message failing is that message's problem. Raised out of here it
            # ended the read loop, and the connection went deaf with nothing said.
            logger.exception("ws: handling %r failed", msg.get("type"))
            try:
                await self.ws.send(json.dumps({
                    "type": "error",
                    "text": f"  ✗ {msg.get('type')} failed: {type(exc).__name__}: {exc}\n"}))
            except Exception:
                logger.debug("handle: the client went away mid-send", exc_info=True)

    async def send_replay(self) -> None:
        """Send what this session produced past the client's watermark, then go live.

        The order in :meth:`run` is the whole argument that this neither duplicates nor
        drops. The subscription was opened *before* the handshake, so everything the
        agents produced since is already queued behind it — there is no gap to fall
        into. The journal is then read from the watermark and sent. Events produced
        while that read was happening are in both places, so the subscription's gate is
        raised to where the replay ended and the drain loop discards anything at or
        below it: the overlap collapses to exactly one copy.

        Streamed deltas are not in the journal and so are not replayed — a resumed turn
        shows the aggregate that closed each block (``thinking_end``, ``answer``) rather
        than its keystrokes, which is also what makes replaying a twelve-hour run
        affordable. ``context_usage``, ``batch_status`` and ``todo`` are re-derived on
        load and are skipped for the same reason.
        """
        session_id = self.view.active_session_id
        if not session_id or self.view.sub is None:
            return
        # Never past what the journal actually holds. The watermark arrives from the
        # client, and is stored, so it can outlive the journal it counted — a session
        # whose log was removed, or a number inherited from another conversation. A
        # claim to have seen more than exists cannot be true, and honouring it sets the
        # gate above every event this session will ever produce.
        #
        # The gate it feeds is this conversation's own, which is what makes the clamp
        # enough: ``seq`` is counted per conversation, so a number from one says
        # nothing about another.
        held = self.pool.bus.last_seq(session_id)
        if self.view.rendered_seq > held:
            logger.warning("replay: session %s claims to have rendered seq %d but its "
                           "journal holds %d; trusting the journal",
                           session_id, self.view.rendered_seq, held)
            self.view.rendered_seq = held
        try:
            events, truncated = read_since(session_id, self.view.rendered_seq,
                                           limit=_REPLAY_MAX_EVENTS)
        except Exception:
            logger.warning("replay: session %s could not be read", session_id,
                           exc_info=True)
            return
        if not events:
            # Nothing missed. The gate still moves to where the journal stands, so a
            # live event this connection has already been handed is not sent twice.
            self.view.sub.rendered_through(session_id, self.view.rendered_seq)
            return

        through = self.view.rendered_seq
        for start in range(0, len(events), _REPLAY_CHUNK):
            chunk = events[start:start + _REPLAY_CHUNK]
            through = max(through, max(int(e.get("seq", 0)) for e in chunk))
            try:
                await self.ws.send(json.dumps({
                    "type": "replay",
                    "session_id": session_id,
                    "events": chunk,
                    "through_seq": through,
                    "more": start + _REPLAY_CHUNK < len(events),
                    "truncated": truncated and start == 0,
                }, default=str))
            except Exception:
                logger.debug("_send_replay: the client went away mid-send", exc_info=True)
                return
        # Only now: a gate raised before the frames were sent would have discarded the
        # live events that arrived between the read and the send.
        self.view.sub.rendered_through(session_id, through)
        logger.info("replay: session %s resent %d event(s) from seq %d",
                    session_id, len(events), self.view.rendered_seq)

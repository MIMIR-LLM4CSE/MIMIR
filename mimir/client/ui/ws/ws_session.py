"""The per-connection WebSocket session for the WS server.

``_Session`` owns one WebSocket connection: which conversation it shows, the drain
loop that forwards the stream to the client, and the inbound-message dispatch table.
The conversation itself is the pool's (``session.conversation``): this view writes
what the user does to it, and renders what the agents did. Split out of
``ws_server.py``; see that module's docstring for the wire protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .event_router import EventRouter
from .transcript_log import read_since
from .ws_pool import _AgentPool
from .ws_worker import _AgentWorker
from .handlers.cards import CardHandlers
from .handlers.conversations import ConversationHandlers
from .handlers.detach import DetachHandlers
from .handlers.panel import PanelHandlers
from .handlers.turns import TurnHandlers
from ...session.conversation import Conversation

# The websocket connection handle passed in by the server (websockets library type
# varies by version); kept as an untyped alias so the annotation resolves.
_WS = Any

logger = logging.getLogger(__name__)


# Events that must reach the user no matter which conversation they belong to: each
# one is a question the agent is parked on, and filtering it as "foreign" (which it is,
# during a background-job wake in another session) would leave the turn waiting on an
# answer the user was never shown. They carry their ``session_id``, so the client can
# say which conversation is asking.
# ``prompt_expired`` rides with them: it takes one of these cards back off the screen,
# and filtering it as foreign would leave a card up for a wait that has ended.
_INTERACTION_EVENTS = frozenset({"approval", "user_question", "prompt_expired"})

# How many journal events one ``replay`` frame carries. The frame cap is 32 MiB, which a
# long detached run's transcript can exceed whole; chunking also lets the webview render
# the first of them while the rest are still arriving.
_REPLAY_CHUNK = 500

# The most a replay will resend. Past this the oldest are dropped and the frame says so:
# what a returning user needs is the end of the run, and an unbounded replay into the
# webview's reducer is how a twelve-hour absence wedges the panel.
_REPLAY_MAX_EVENTS = 5000

class _Session(ConversationHandlers, TurnHandlers, CardHandlers, DetachHandlers,
               PanelHandlers):
    """One WebSocket connection — owns a chat history and talks to the worker."""

    def __init__(self, ws: "_WS", pool: "_AgentPool") -> None:
        self.ws = ws
        self.pool = pool
        # The conversation on screen — the pool's own object for it, shared with
        # everything else that writes it. This connection only shows it.
        self.conv: Conversation | None = None
        self.store = pool.conversations.store
        self._active_session_id: str | None = None
        # In-flight session-summary refresh (one at a time per connection).
        self._summary_task: asyncio.Task | None = None
        # Last `used_tokens` pushed to the client, so the periodic mid-turn refresh
        # only sends a frame when the number actually moved.
        self._last_context_usage: int | None = None
        # This connection's view of the stream, opened in run(). The pump owns the
        # draining; this is only where our share of it arrives.
        self._sub = None
        # Events withheld because they belong to another conversation, per session.
        # The one quiet-chat cause that is otherwise invisible — see _is_foreign_event.
        self._foreign_withheld: dict[str, int] = {}
        # How far into the journal the client's rendered transcript goes, as the client
        # last said. The watermark a re-attach replays from.
        self._rendered_seq = 0
        # Tool rows of the running turn that publish a run channel, as
        # {call id: tool name}. The name IS the channel (see run_channel), so this is
        # what lets a divert click name the run the user is actually looking at, and
        # what the progress poller walks. Registry data passing through: nothing here
        # compares a name against a literal.
        # Keyed by session: with turns running concurrently, two conversations can each
        # have a divertible run in flight, and one map between them would let a divert
        # click detach the other one's command.
        self._live_rows: dict[str, dict[str, str]] = {}
        # Last (phase, percent) pushed per call id, so a tick that learned nothing
        # sends nothing.
        self._sent_progress: dict[str, dict[str, tuple]] = {}
        # Ids of deferred cards the user moved past without answering. Their card may
        # still be on screen; an answer to one must not reach the next live prompt.
        self._stale_prompt_ids: set[str] = set()
        # Sessions whose deletion was refused once because work of theirs was still
        # running. Asking again goes through: the warning is there to be read, not to
        # make the conversation undeletable, and the second click is the confirmation.
        self._delete_refused: set[str] = set()
        # The panel report in flight, and whether another ask (and a re-read) came in
        # behind it. Asks arriving while one is served are folded into one more pass.
        self._panel_task: asyncio.Task | None = None
        self._panel_again: dict | None = None

    @property
    def worker(self) -> _AgentWorker:
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
        return self.pool.worker_or_detached(self._active_session_id)

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
        return {"type": "ready", **self.pool.ui_state(self._active_session_id)}

    async def run(self) -> None:
        # Subscribe BEFORE anything else, and this order is the point: from here on
        # nothing a turn produces can be missed, because it accumulates in this
        # subscription's queue while the rest of the handshake runs. Nothing has to be
        # swept either — the pump has already drained and journaled everything, so what
        # a worker holds is never debris.
        self._sub = self.pool.bus.subscribe()
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
            self._purge_empty_sessions()
            await self._send_sessions_list()
            await self._send_toggles()

            sessions = self.store.list_sessions()
            if sessions:
                # Auto-load the most recent session.
                try:
                    await self._load_session(sessions[0].id)
                except Exception:
                    await self._create_new_session()
            else:
                await self._create_new_session()

            # What happened while nobody was attached, before the cards: a replay under a
            # chat that is already on screen, and a card on top of that.
            await self._send_replay()

            # The run never stopped, so its own level is the truth the panel must show.
            await self._restore_detached_autonomy()

            # Last, so the card lands under a chat that is already on screen.
            await self._resend_parked_prompt()

            await self._send_served_models()

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
            jobs_task = asyncio.create_task(self._report_ended_jobs())
            try:
                async for raw in self.ws:
                    await self._handle(raw)
            except Exception:
                logger.debug("run: a best-effort step failed and was skipped", exc_info=True)
        finally:
            # Asked for rather than read: this block runs on every way out, including
            # a greeting that never reached the socket, and cleanup that raises here
            # strands the subscription it is on its way to close.
            summary_task = getattr(self, "_summary_task", None)
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
            self._sub.close()

    async def _drain_loop(self) -> None:
        """Forward this connection's share of the stream to the client (``EventRouter``).

        ``run()`` subscribes before the handshake so nothing can be missed; the router
        subscribes itself for a caller that starts the loop on its own.
        """
        await EventRouter(self).run()

    # Maps an inbound WS message "type" to the _Session handler method that serves it.
    # Replaces the former 13-branch ``if mtype == …`` chain in _handle.

    # Inbound message type -> the method that serves it, from every handler mixin.
    _MSG_HANDLERS: dict[str, str] = {
        **ConversationHandlers.MESSAGES, **TurnHandlers.MESSAGES, **CardHandlers.MESSAGES,
        **DetachHandlers.MESSAGES, **PanelHandlers.MESSAGES,
    }

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        handler_name = self._MSG_HANDLERS.get(msg.get("type"))
        if handler_name is None:
            return  # unknown/missing type — ignore, as the old fall-through did
        try:
            await getattr(self, handler_name)(msg)
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
                logger.debug("_handle: the client went away mid-send", exc_info=True)

    async def _send_replay(self) -> None:
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
        session_id = self._active_session_id
        if not session_id or self._sub is None:
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
        if self._rendered_seq > held:
            logger.warning("replay: session %s claims to have rendered seq %d but its "
                           "journal holds %d; trusting the journal",
                           session_id, self._rendered_seq, held)
            self._rendered_seq = held
        try:
            events, truncated = read_since(session_id, self._rendered_seq,
                                           limit=_REPLAY_MAX_EVENTS)
        except Exception:
            logger.warning("replay: session %s could not be read", session_id,
                           exc_info=True)
            return
        if not events:
            # Nothing missed. The gate still moves to where the journal stands, so a
            # live event this connection has already been handed is not sent twice.
            self._sub.rendered_through(session_id, self._rendered_seq)
            return

        through = self._rendered_seq
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
        self._sub.rendered_through(session_id, through)
        logger.info("replay: session %s resent %d event(s) from seq %d",
                    session_id, len(events), self._rendered_seq)

    async def _notify(self, text: str) -> None:
        """One line of chrome for the client, outside any conversation. Never raises."""
        try:
            await self.ws.send(json.dumps({"type": "output", "text": f"  {text}\n"}))
        except Exception:
            logger.debug("_notify: the client went away mid-send", exc_info=True)

    def _apply_setting(self, name: str, value: Any) -> str:
        """Apply a UI knob to every live conversation, and to the next agent built.

        Both halves are needed. Pushing it to the live workers makes the change take effect
        now; recording it in the pool makes a conversation started afterwards come up on it
        rather than on the default, which matters because an agent is built long after the
        user sets these.

        Returns the first rejection a live worker reported, or "". Nothing to reject when
        none exists yet: the setting is recorded and validated when it is replayed.
        """
        return next(iter(self.pool.apply_setting(name, value)), "")

    def _schedule(self, coro: Any) -> None:
        """Fire a send without awaiting it, for callers that are not coroutines."""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()

    def _is_foreign_event(self, ev: dict) -> bool:
        """True when *ev* was produced for a session other than the active one.

        Every event of a running turn is stamped with the session it was produced for.
        Several conversations can be producing at once, and only one of them is on screen,
        so the stamp is what keeps one conversation's output out of another's chat — it goes
        to that conversation's own transcript instead. Unstamped events (produced outside
        any conversation) always pass.

        So do the interaction events. Each one is a question the agent is parked on,
        and a background-job wake runs turns in sessions the user is not looking at:
        filtering those prompts as foreign would park the turn forever on an answer
        nobody was shown. They carry their session id, so the prompt can say which
        conversation is asking.
        """
        if ev.get("type") in _INTERACTION_EVENTS:
            return False
        ev_session = ev.get("session_id")
        return ev_session is not None and ev_session != self._active_session_id

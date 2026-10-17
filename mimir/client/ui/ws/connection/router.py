"""One connection's share of the stream, forwarded to its client.

The pump has already done the work by the time an event arrives here: it journaled it,
wrote an answer into its conversation, routed a finished run to the turn it is owed.
What is left is the view: send the event, keep the bookkeeping a screen needs (the
rows a divert click or a progress bar resolves through, the context bar, the review
bar), and say what the pool did that the client cannot infer.

It waits on its subscription rather than polling it, so an idle connection costs
nothing: the loop wakes for an event, and for the once-a-second refreshes only while
a turn of the conversation on screen is live.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Awaitable, Callable

from ..runtime import todo_file_for_session
from ....protocol.events import DURABLE_EVENTS
from ....tool_execution import run_channel

if TYPE_CHECKING:
    from .connection import Connection

logger = logging.getLogger(__name__)

# How often a live turn's context bar and run progress are refreshed.
_TICK = 1.0


class _SocketGone(Exception):
    """The client went away mid-send. The record is the journal, so nothing is lost."""


class EventRouter:
    """Forwards a session's subscription to its socket, one handler per event type."""

    def __init__(self, session: "Connection") -> None:
        self.session = session
        self._todo_mtime: float | None = None
        self._last_tick = 0.0
        # What to do once an event has been sent, by type.
        self._after: dict[str, Callable[[dict, dict, str | None], Awaitable[None]]] = {
            "tool_call": self._after_tool_call,
            "tool_result": self._after_tool_result,
            "error": self._after_error,
            "file_progress": self._after_file_progress,
            "diff": self._after_diff,
            "answer": self._after_answer,
        }

    async def run(self) -> None:
        """Until the socket goes. Every event of the subscription, in order."""
        session = self.session
        if session._sub is None:
            session._sub = session.pool.bus.subscribe()
        queue = session._sub.queue
        try:
            while True:
                item = await self._next(queue)
                batch = [] if item is None else [item]
                while True:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                for ev, extras in batch:
                    await self.route(ev, extras)
                await self._tick()
        except _SocketGone:
            return

    async def _next(self, queue: asyncio.Queue) -> tuple[dict, dict] | None:
        """The next event — or None after a tick's wait, when there are refreshes to do."""
        if not self._live():
            return await queue.get()
        try:
            return await asyncio.wait_for(queue.get(), _TICK)
        except asyncio.TimeoutError:
            return None

    def _live(self) -> bool:
        """Whether the conversation on screen has anything a refresh would show."""
        session = self.session
        return (session._running_turn_is_ours()
                or bool(session._rows_for(session._active_session_id)))

    async def _send(self, payload: dict) -> None:
        try:
            await self.session.ws.send(json.dumps(payload, default=str))
        except Exception as exc:
            raise _SocketGone from exc

    # ── Routing ───────────────────────────────────────────────────────────────

    async def route(self, ev: dict, extras: dict) -> None:
        session = self.session
        # Which conversation this belongs to. Everything below addresses a session
        # rather than "the session", because more than one may be producing at once.
        owner = ev.get("session_id") or session._active_session_id
        etype = ev.get("type")
        if etype in DURABLE_EVENTS:
            await self._durable(ev, extras)
            return
        if session._is_foreign_event(ev):
            await self._foreign(ev, extras, owner)
            return
        ev = self._unwrapped(ev)
        # Already journaled — the bus records before it fans out — so a send that fails
        # loses the socket, never the record.
        await self._send(ev)
        after = self._after.get(ev.get("type", ""))
        if after is not None:
            await after(ev, extras, owner)

    @staticmethod
    def _unwrapped(ev: dict) -> dict:
        """An event a tool printed as a JSON line (a diff, say) is that event."""
        if ev.get("type") != "output":
            return ev
        text = ev.get("text", "").strip()
        if text.startswith("{") and text.endswith("}"):
            try:
                inner = json.loads(text)
            except json.JSONDecodeError:
                return ev
            if isinstance(inner, dict) and "type" in inner:
                return inner
        return ev

    async def _durable(self, ev: dict, extras: dict) -> None:
        """A run finished, or a bulletin on runs still going — already routed by the
        pool to the conversation that launched them. What is left is to say so.

        A wake starts a turn nobody pressed send for, and the client marks itself busy
        only on its own submit — so it is told, and it cannot work it out for itself:
        the wake may resume a conversation other than the one on screen, and marking
        the visible chat busy for a turn running elsewhere leaves a stop button that
        stops nothing. A run steered into the turn already running starts nothing, and
        saying it did makes the client drop that turn's live rows.
        """
        session = self.session
        started = bool(extras.get("_started"))
        await self._send({**ev, "resumes_active_session":
                          started and not session._is_foreign_event(ev)})
        if ev.get("type") != "job_checkin":
            # One run fewer. Behind the notification: the news is what the user is
            # waiting for, and this is bookkeeping.
            await session._send_sessions_list()

    async def _foreign(self, ev: dict, extras: dict, owner: str | None) -> None:
        """An event of a conversation that is not on screen: not drawn here.

        It still happened, and it is in that conversation's own journal, so switching
        to it shows the whole turn. Counted, because a chat gone quiet this way leaves
        no other trace from the outside.
        """
        session = self.session
        if owner:
            session._foreign_withheld[owner] = session._foreign_withheld.get(owner, 0) + 1
        if ev.get("type") == "answer":
            await session._announce_detached_answer(extras)
        if ev.get("type") in ("answer", "error"):
            await session._admit_waiting()

    # ── After the send ────────────────────────────────────────────────────────

    async def _after_tool_call(self, ev: dict, extras: dict, owner: str | None) -> None:
        # Only divertible rows: they are exactly the ones a channel can answer for, and
        # the map is what both the divert click and the progress refresh resolve through.
        if ev.get("divertible"):
            self.session._rows_for(owner)[str(ev.get("id") or "")] = str(ev.get("name") or "")

    async def _after_tool_result(self, ev: dict, extras: dict, owner: str | None) -> None:
        self.session._forget_row(str(ev.get("id") or ""), owner)
        # A tool is what writes the checklist, so a result is when it may have changed.
        await self._push_todos()

    async def _after_error(self, ev: dict, extras: dict, owner: str | None) -> None:
        # A turn failed (often a context-overflow 400) and no answer follows: its rows
        # are over, and the bar must show the failure, not the usage before it.
        self._turn_over(owner)
        await self.session._emit_context_usage()
        await self.session._admit_waiting()

    async def _after_file_progress(self, ev: dict, extras: dict, owner: str | None) -> None:
        # The review bar appears and grows mid-turn, not only once the agent finishes.
        await self._push_batch_status(only_if_any=True)
        # So the browser can draw the "writing…" card before the write completes and the
        # diff and answer arrive on top of it.
        await asyncio.sleep(0.1)

    async def _after_diff(self, ev: dict, extras: dict, owner: str | None) -> None:
        await self._push_batch_status(only_if_any=True)

    async def _after_answer(self, ev: dict, extras: dict, owner: str | None) -> None:
        """The turn landed. The pool has written it, and started whatever it owed."""
        session = self.session
        self._turn_over(owner)
        if extras.get("_deferred"):
            # Set aside while this conversation was on screen: its card comes back.
            await session._resend_deferred_prompt()
        # Now, so the description says what was done rather than what was asked, and
        # its model call no longer competes with the query the user is waiting on.
        session._schedule_summary_refresh()
        await session._emit_context_usage()
        # This conversation stopping may let the pool free a slot for one waiting.
        await session._admit_waiting()
        # What it is still doing, now that it has stopped answering: a turn that
        # launched a background run ends with none in flight and a run going, and the
        # panel decides from this whether walking away would abandon work.
        await session._send_sessions_list()
        # Read now rather than queued by the worker, so a review accepted mid-run has
        # already cleared what this reports.
        await self._push_batch_status(only_if_any=False)
        await self._push_todos()
        committed = extras.get("_committed") or {}
        # Steering past the last step boundary was never read, and the pool gave it a
        # turn of its own. Each bubble loses its "queued" tag, and the last says a turn
        # started: without it the catch-up turn runs with no stop button.
        await session._announce_steer_turns(committed.get("steer_turn") or [])

    def _turn_over(self, owner: str | None) -> None:
        """No row of the finished turn is still running. Its own conversation's rows —
        another turn may be running in another one, and its rows are live."""
        self.session._rows_for(owner).clear()
        self.session._sent_progress.setdefault(owner or "", {}).clear()

    async def _push_batch_status(self, *, only_if_any: bool) -> None:
        try:
            files = self.session.worker.review.batch_status()
        except Exception:
            logger.debug("batch status could not be built", exc_info=True)
            return
        if files or not only_if_any:
            await self._send({"type": "batch_status", "files": files})

    async def _push_todos(self) -> None:
        """Send the checklist of the conversation on screen, if its file changed."""
        session = self.session
        try:
            path = todo_file_for_session(session._active_session_id)
            mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
        except OSError:
            return
        if mtime == self._todo_mtime:
            return
        self._todo_mtime = mtime
        try:
            items = session.worker.review.todos()
        except Exception:
            logger.debug("checklist could not be read", exc_info=True)
            return
        if items:
            await self._send({"type": "todo", "items": items})

    # ── Refreshes while a turn is live ────────────────────────────────────────

    async def _tick(self) -> None:
        """At most once a second, while a turn of the conversation on screen is live."""
        now = time.monotonic()
        if not self._live() or now - self._last_tick < _TICK:
            return
        self._last_tick = now
        session = self.session
        if session._running_turn_is_ours():
            # A turn can run dozens of tool calls over minutes; without this the bar
            # only moves when the answer lands. A wake turn of another conversation is
            # not this one's usage.
            await session._emit_context_usage()
        await self._push_run_progress()

    async def _push_run_progress(self) -> None:
        """What each blocking run of the conversation on screen is doing.

        A run that blocks the turn for twenty minutes shows a spinner and nothing else;
        the server publishes its phase on the run channel precisely because the call
        it belongs to cannot answer until it is over. Transient — not recorded, like
        the context bar: it describes a moment.
        """
        session = self.session
        sid = session._active_session_id
        rows = session._rows_for(sid)
        if not rows:
            return
        sent = session._sent_progress.setdefault(sid or "", {})
        names = list(rows.values())
        for call_id, name in list(rows.items()):
            # A name serving two running rows cannot be attributed to either of them.
            if names.count(name) > 1:
                continue
            try:
                run = run_channel.current_run(name, sid)
            except Exception:
                logger.debug("_push_run_progress: a best-effort step failed and was skipped", exc_info=True)
                continue
            if not run:
                continue
            phase = str(run.get("phase") or "")
            percent = run.get("percent")
            if not isinstance(percent, (int, float)):
                percent = None
            # Skipped only while nothing changed: once a bar is up, its retraction is a
            # change the row must hear about, or the bar stays frozen.
            if sent.get(call_id, ("", None)) == (phase, percent):
                continue
            sent[call_id] = (phase, percent)
            await self._send({"type": "tool_progress", "id": call_id,
                              "phase": phase, "percent": percent})

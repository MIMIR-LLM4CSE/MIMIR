"""The event bus: one pump for the whole process, the journal as the record.

The journal is the bus and a socket is one of its followers. The pump lives for the
process, drains every worker whether or not anyone is listening, writes each event to
that session's ``transcript.jsonl`` under a monotonic ``seq``, and only then fans it out
to however many sockets happen to be subscribed — zero included:

    worker.out_q ──(pump)──► seq ──► transcript.jsonl      (durable, authoritative)
                               │
                               ├──► the pool               (the conversation: an answer
                               │                            written, a run's wake routed)
                               └──► 0..N subscriber queues ──► sockets

Anything that recorded a turn from inside a connection would stop recording when the
connection dropped: the agents keep working — the pool is process-global, and
``releasable`` refuses to evict a worker that is busy — so what they do has to be
written by something that outlives every socket. Three properties follow, and they are
the whole reason for the module:

**``out_q`` is bounded by a pump tick** rather than by how long the user was away. The
queue stays unbounded upstream — the worker thread must never block on a full queue
mid-turn — but it is now always drained.

**The journal is authoritative, so a subscriber may drop events.** A slow or absent
client cannot hold the process hostage: its queue is bounded, and overflow is reported
as a gap the client closes by re-reading the journal from its watermark. That is the
same path as a fresh attach, not a special case.

**``seq`` is assigned here and nowhere else.** One writer per session is what makes the
number a watermark a client can resume from. Anything else that appends to a session's
journal must come through :meth:`record_client_event`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Iterable

from ...protocol.events import (
    CONCLUSIVE_EVENTS, DURABLE_EVENTS, REPLAY_ONLY_EVENTS, pop_answer_extras,
)
from ...session.journal import TranscriptLog

logger = logging.getLogger(__name__)

# How often the pump looks without being asked. A worker's outbox nudges it on every
# event (see ``nudge``), so this is not the latency of anything a turn produces: it is
# the housekeeping tick — telling parked workers whether anyone is attached — and the
# backstop for an event put before its worker was wired to nudge.
_PUMP_INTERVAL = 1.0

# How many events one subscriber may fall behind before it is told it has a gap. Past
# this the oldest are dropped: the journal still holds them, so the client recovers by
# replaying from its watermark rather than by the pump stalling for it.
_SUBSCRIBER_MAX = 2000

class _Subscription:
    """One attached client's view of the stream.

    Bounded on purpose. The queue is a delivery buffer, not a record: the record is the
    journal, and a client that cannot keep up is better told about a gap than allowed
    to make the server hold every event it has not read.
    """

    def __init__(self, bus: "_EventBus", session_filter: Callable[[dict], bool] | None,
                 maxsize: int = _SUBSCRIBER_MAX) -> None:
        self._bus = bus
        self._filter = session_filter
        # ``(event, extras)`` pairs. The extras are the session-layer payload the pump
        # takes off an answer: they must not reach the journal or the wire, but the
        # session that owns the turn still needs them to write its history — so they
        # travel beside the event rather than inside it.
        self.queue: asyncio.Queue[tuple[dict, dict]] = asyncio.Queue(maxsize=maxsize)
        # Per conversation, the seq at or below which this client has already rendered
        # everything: the replay it was just sent covers them. Written by the attach
        # path, one conversation at a time.
        #
        # Per conversation and not one number for the socket, because ``seq`` is
        # counted per conversation — one journal and one writer each. A single
        # watermark raised to one conversation's position gates out every conversation
        # whose journal is shorter, which on a socket that opens on a long chat is
        # every new one: its events start again at 1, below the gate, and are dropped.
        # Streamed text is never journaled and so passes regardless, which is what
        # makes the failure look like a working chat that has stopped showing tool
        # calls, diffs and answers rather than like a dead connection.
        self._rendered: dict[str, int] = {}
        # Set when the queue overflowed. The client is told, and closes the gap by
        # replaying from its own watermark.
        self.gapped: bool = False
        self.dropped: int = 0
        # Events the watermark filtered out. Counted and logged rather than silently
        # discarded: a gate set too high is invisible from the outside.
        self.filtered: int = 0

    def rendered_through(self, session_id: str, seq: int) -> None:
        """Note that *session_id* is rendered up to *seq* on this client.

        Monotonic per conversation: a gate only ever rises, so an event already handed
        to this connection is not sent again by a later replay of the same chat.
        """
        if not session_id:
            return
        self._rendered[session_id] = max(self._rendered.get(session_id, 0), int(seq))

    def rendered_at(self, session_id: str) -> int:
        """The watermark for one conversation, or 0 if it has none."""
        return self._rendered.get(session_id or "", 0)

    def wants(self, ev: dict) -> bool:
        if self._filter is not None and not self._filter(ev):
            return False
        seq = ev.get("seq")
        min_seq = self._rendered.get(ev.get("session_id") or "", 0)
        # Unstamped events are stream deltas: live-only, never replayed, so the
        # watermark has nothing to say about them. Note the asymmetry this creates and
        # why it is worth logging: `token` and `thinking` are never journaled, so they
        # bypass the gate entirely. A gate set too high therefore does not look like a
        # dead connection — it looks like a chat that streams text and reasoning
        # normally and never shows a tool call, a diff or an answer.
        if isinstance(seq, int) and seq <= min_seq:
            self.filtered += 1
            if self.filtered == 1 or self.filtered % 100 == 0:
                logger.warning("subscription: dropped %d event(s) at or below the "
                               "rendered watermark %d of session %s (latest: %s seq "
                               "%d)", self.filtered, min_seq, ev.get("session_id"),
                               ev.get("type"), seq)
            return False
        return True

    def offer(self, ev: dict, extras: dict | None = None) -> None:
        if not self.wants(ev):
            return
        item = (ev, extras or {})
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop the oldest, not the newest: what a client needs most is the end of
            # the stream, and the beginning is what the journal replays best.
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except asyncio.QueueEmpty:
                pass
            self.gapped = True
            try:
                self.queue.put_nowait(item)
            except asyncio.QueueFull:
                self.dropped += 1

    def close(self) -> None:
        self._bus.unsubscribe(self)


class _EventBus:
    """Drains every worker, journals what they emit, and fans it out.

    Owned by the pool, started by the server, and alive for the process — which is the
    point: it must keep recording when no connection exists.
    """

    def __init__(self, pool, *, commit: Callable[[dict, dict], dict] | None = None,
                 durable: Callable[[dict], bool] | None = None,
                 steered: Callable[[str | None, str], None] | None = None,
                 interval: float = _PUMP_INTERVAL) -> None:
        self._pool = pool
        # What the pump hands an event to before any socket sees it: the answer to the
        # conversation it ends, a finished run's wake to the conversation that launched
        # it, the loop's receipt for a steer to the runs it named. Before the fan-out,
        # and whether or not anyone is attached: a subscriber is a view, and what a turn
        # produces has to be acted on with nobody looking.
        self._commit = commit
        self._durable = durable
        self._steered = steered
        self._interval = interval
        self._logs: dict[str, TranscriptLog] = {}
        self._subs: list[_Subscription] = []
        # When each session last *concluded* — delivered a final answer, or ended in an
        # error. Not "when it was last heard from": an idle-shutdown predicate built on
        # the absence of noise would stop a server whose turn merely paused, so the
        # criterion is positive. Cleared the moment the session produces again, which is
        # what restarts the clock rather than letting a stale conclusion stand.
        self._concluded: dict[str, float] = {}
        # When the last client left, or None while one is here. Pushed onto every
        # worker each tick, because the thread that needs to know — the one parked on
        # an approval — must not read loop state to find out.
        self._unattended_since: float | None = time.monotonic()
        # Whether a client has ever been here. "Its last client left" presupposes one:
        # a server started by hand and not yet connected to has lost nothing, and must
        # not stop itself out from under the window that is about to attach.
        self.ever_attached = False
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        # Counters, so the chain can be asked what it did rather than inferred from
        # what the chat shows. The pump is the only consumer of every worker's queue, so
        # "the chat went quiet" has several causes that look identical from outside: the
        # pump never started, it started and died, it is draining but a watermark is
        # swallowing the result, or the agent emitted nothing. These tell them apart in
        # one answer.
        self.ticks = 0
        self.journaled = 0
        self.fanned_out = 0
        self.pump_errors = 0
        self.last_error: str = ""

    # ── Subscribers ───────────────────────────────────────────────────────────

    def subscribe(self, session_filter: Callable[[dict], bool] | None = None
                  ) -> _Subscription:
        sub = _Subscription(self, session_filter)
        self._subs.append(sub)
        self.ever_attached = True
        return sub

    def unsubscribe(self, sub: _Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    def attached(self, *, excluding: _Subscription | None = None) -> int:
        """How many clients are listening. Zero is the detached case.

        *excluding* leaves one subscription out, for the question a window asks about
        its own departure: it is itself an attachment, so counting it answers "yes,
        somebody is watching — you".
        """
        if excluding is None:
            return len(self._subs)
        return sum(1 for sub in self._subs if sub is not excluding)

    def unattended_for(self) -> float | None:
        """Seconds since the last client left, or None while one is here.

        None is "somebody is watching". A number is how long nobody has been — which
        is what decides whether a server nobody asked to keep has any reason to live.
        """
        since = self._unattended_since
        return None if since is None else max(0.0, time.monotonic() - since)

    def diagnostics(self) -> list[dict]:
        """What each link of the chain has actually done, as {label, detail} rows.

        For the one question that is hard to answer from the outside: the chat has gone
        quiet, and the pump never starting, the pump dying, a watermark swallowing
        everything and the agent emitting nothing all look the same from a chat window.
        """
        running = self._task is not None and not self._task.done()
        rows = [
            {"label": "pump", "detail":
                f"{'running' if running else 'NOT RUNNING'} — {self.ticks} tick(s)"},
            {"label": "journaled", "detail": f"{self.journaled} event(s) stamped"},
            {"label": "fanned out", "detail": f"{self.fanned_out} delivery attempt(s)"},
            {"label": "subscriptions", "detail": str(len(self._subs))},
            {"label": "unattended for",
             "detail": "attached" if self._unattended_since is None
                       else f"{time.monotonic() - self._unattended_since:.0f}s"},
        ]
        if self.pump_errors:
            rows.append({"label": "pump errors",
                         "detail": f"{self.pump_errors} — last: {self.last_error}"})
        for index, sub in enumerate(self._subs, start=1):
            rows.append({"label": f"subscription {index}", "detail":
                         f"watermarks {sub._rendered or '{}'}, "
                         f"{sub.filtered} filtered, "
                         f"{sub.dropped} dropped, {sub.queue.qsize()} queued"})
        for session_id, log in self._logs.items():
            rows.append({"label": f"journal {session_id[:8]}",
                         "detail": f"seq {log.seq}"})
        return rows

    # ── What counts as finished ───────────────────────────────────────────────

    def concluded_at(self, session_id: str) -> float | None:
        """When this session last delivered a final answer, or None if it has not.

        None is the answer for every state that is not a conclusion — a turn in
        flight, a turn parked on a card, a worker that has only just been built, a turn
        that was cancelled. A shutdown predicate must treat all of those as *working*,
        because none of them is a session that finished.
        """
        return self._concluded.get(session_id)

    def concluded_since(self, session_id: str, now: float | None = None) -> float | None:
        """Seconds since this session concluded, or None if it has not concluded."""
        at = self._concluded.get(session_id)
        if at is None:
            return None
        return max(0.0, (time.time() if now is None else now) - at)

    # ── The journal ───────────────────────────────────────────────────────────

    def _log_for(self, session_id: str) -> TranscriptLog:
        """This session's writer, kept so its ``seq`` is not re-read from disk per event.

        ``TranscriptLog.bind`` scans the file for the highest ``seq`` already written,
        which is right once per session and ruinous per event.
        """
        log = self._logs.get(session_id)
        if log is None:
            log = TranscriptLog()
            log.bind(session_id)
            self._logs[session_id] = log
        return log

    def record_client_event(self, session_id: str, ev: dict) -> dict:
        """Journal something the *client* said (a query, a steer) under the same seq run.

        The journal has one writer per session; a second one interleaves the numbering
        and breaks every watermark derived from it. So a line the session layer wants
        recorded comes through here rather than being appended directly.
        """
        if not session_id:
            return ev
        try:
            return self._log_for(session_id).append_and_stamp(ev)
        except Exception:
            logger.warning("bus: could not journal a client event for %s", session_id,
                           exc_info=True)
            return ev

    def last_seq(self, session_id: str) -> int:
        """The highest ``seq`` written for this session."""
        if not session_id:
            return 0
        try:
            return self._log_for(session_id).seq
        except Exception:
            logger.debug("last_seq: a best-effort step failed and was skipped", exc_info=True)
            return 0

    # ── The pump ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.get_running_loop().create_task(self._pump_loop())

    async def aclose(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    def nudge(self) -> None:
        """Ask the pump to look now rather than at the next tick. On the loop only — a
        worker thread goes through ``call_soon_threadsafe`` (see ``_Outbox``)."""
        try:
            self._wake.set()
        except Exception:
            logger.debug("nudge: a best-effort step failed and was skipped", exc_info=True)

    async def _pump_loop(self) -> None:
        while True:
            # Cleared before the tick, not after the wait: a nudge that arrives while
            # this tick runs is then the next wake-up rather than a lost one.
            self._wake.clear()
            try:
                self.pump_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # A pump that dies stops recording every session at once, which is the
                # one failure this module exists to prevent. Log and keep going.
                self.pump_errors += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("bus: pump tick failed", exc_info=True)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise

    def _publish_attachment(self) -> None:
        """Tell every worker how long this process has been without a client.

        A worker parked on a card is a *thread*, blocked in a queue poll. It cannot
        await anything and must not touch the loop's state, so the answer is pushed to
        it as a plain timestamp rather than asked for. Pushed every tick, including the
        ticks that move no events, since the thing it reports is the absence of events.
        """
        if self._subs:
            self._unattended_since = None
        elif self._unattended_since is None:
            self._unattended_since = time.monotonic()
        try:
            workers = list(self._pool.items())
        except Exception:
            logger.debug("_publish_attachment: a best-effort step failed and was skipped", exc_info=True)
            return
        for _sid, worker in workers:
            try:
                worker.human.unattended_since = self._unattended_since
            except Exception:
                logger.debug("_publish_attachment: a best-effort step failed and was skipped", exc_info=True)

    def pump_once(self) -> int:
        """Drain, journal and fan out one round. Returns how many events moved.

        Separate from the loop so a test can run exactly one tick, and so the shutdown
        path can flush what is still queued before the process goes.
        """
        self.ticks += 1
        self._publish_attachment()
        moved = 0
        for ev in self._collect():
            owner = ev.get("session_id") or ""
            extras = pop_answer_extras(ev)
            if owner:
                try:
                    ev = self._log_for(owner).append_and_stamp(ev)
                except Exception:
                    logger.warning("bus: could not journal an event for %s", owner,
                                   exc_info=True)
            etype = ev.get("type")
            if owner:
                if etype in CONCLUSIVE_EVENTS:
                    self._concluded[owner] = time.time()
                else:
                    # It is producing again, so whatever it concluded before is history.
                    self._concluded.pop(owner, None)
            # After the journal, so what the pool acts on is the stamped event, and
            # before the fan-out, so what a socket renders is already written: the
            # conversation holds the answer, the run's turn has started. What the pool
            # did travels to the sockets in the extras.
            self._hand_over(ev, extras)
            if isinstance(ev.get("seq"), int):
                self.journaled += 1
            if etype in REPLAY_ONLY_EVENTS:
                continue   # journaled above; the live client had the deltas
            for sub in list(self._subs):
                sub.offer(ev, extras)
                self.fanned_out += 1
            moved += 1
        return moved

    def _hand_over(self, ev: dict, extras: dict) -> None:
        """Let the pool act on *ev*, putting what it did in *extras*. Never raises: one
        conversation that cannot be written must not stop the tick draining the rest."""
        etype = ev.get("type")
        try:
            if etype == "answer" and self._commit is not None:
                extras["_committed"] = self._commit(ev, extras)
            elif etype in DURABLE_EVENTS and self._durable is not None:
                extras["_started"] = bool(self._durable(ev))
            elif etype == "steer_injected" and self._steered is not None:
                self._steered(ev.get("session_id"), str(ev.get("text") or ""))
        except Exception:
            logger.warning("bus: handling %s for %s failed", etype, ev.get("session_id"),
                           exc_info=True)

    def _collect(self) -> Iterable[dict]:
        """Every pending event, across every live worker.

        Per-conversation order comes free — each worker owns its own FIFO — and the
        interleaving between conversations is harmless because every event carries the
        session it was produced for.
        """
        events: list[dict] = []
        try:
            workers = list(self._pool.items())
        except Exception:
            logger.debug("_collect: a best-effort step failed and was skipped", exc_info=True)
            return events
        for _sid, worker in workers:
            try:
                events.extend(worker.drain())
            except Exception:
                logger.warning("bus: could not drain a worker", exc_info=True)
        return events

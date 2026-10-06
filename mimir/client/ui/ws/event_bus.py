"""The event bus: one pump for the whole process, the journal as the record.

Without this the WebSocket *is* the bus. ``_Session._drain_loop`` is created per
connection, and it is what calls ``worker.drain()``, tees the journal and writes a
finished turn's answer into its session file — so a socket that drops takes all three
with it. The agents keep working (the pool is process-global, and ``releasable``
refuses to evict a worker that is busy), but nothing records what they do, ``out_q``
grows for the length of the disconnection, and a turn that finishes while nobody is
attached loses its answer.

So the arrow is turned around. The pump lives for the process, drains every worker
whether or not anyone is listening, writes each event to that session's
``transcript.jsonl`` under a monotonic ``seq``, and only then fans it out to however
many sockets happen to be subscribed — zero included:

    worker.out_q ──(pump)──► seq ──► transcript.jsonl      (durable, authoritative)
                               │
                               ├──► turn committer         (the session file)
                               └──► 0..N subscriber queues ──► sockets

Three properties follow, and they are the whole reason for the module:

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

from .transcript_log import TranscriptLog

logger = logging.getLogger(__name__)

# How often the pump looks for output. Fast enough that streamed tokens do not read as
# stutter, slow enough that an idle server is not a busy loop. The pump also wakes on
# demand (see ``nudge``), so this is the ceiling on latency, not the usual case.
_PUMP_INTERVAL = 0.05

# How many events one subscriber may fall behind before it is told it has a gap. Past
# this the oldest are dropped: the journal still holds them, so the client recovers by
# replaying from its watermark rather than by the pump stalling for it.
_SUBSCRIBER_MAX = 2000

# The event types that end a turn. An ``error`` is a conclusion as much as an ``answer``
# is — the turn is over either way — and conflating "concluded" with "succeeded" would
# leave a session that failed looking busy for ever.
_CONCLUSIVE = frozenset({"answer", "error"})


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
        # Events at or below this are already rendered by this client: the replay it
        # was just sent covers them. Set once, after the replay, by the attach path.
        self.min_seq: int = 0
        # Set when the queue overflowed. The client is told, and closes the gap by
        # replaying from its own watermark.
        self.gapped: bool = False
        self.dropped: int = 0

    def wants(self, ev: dict) -> bool:
        if self._filter is not None and not self._filter(ev):
            return False
        seq = ev.get("seq")
        # Unstamped events are stream deltas: live-only, never replayed, so the
        # watermark has nothing to say about them.
        if isinstance(seq, int) and seq <= self.min_seq:
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

    def __init__(self, pool, *, commit: Callable[[dict, dict], None] | None = None,
                 interval: float = _PUMP_INTERVAL) -> None:
        self._pool = pool
        self._commit = commit
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
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()

    # ── Subscribers ───────────────────────────────────────────────────────────

    def subscribe(self, session_filter: Callable[[dict], bool] | None = None
                  ) -> _Subscription:
        sub = _Subscription(self, session_filter)
        self._subs.append(sub)
        return sub

    def unsubscribe(self, sub: _Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    def attached(self) -> int:
        """How many clients are listening. Zero is the detached case."""
        return len(self._subs)

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

        The journal has one writer per session; a second one would interleave the
        numbering and break every watermark derived from it. So the lines that used to
        be appended by the session layer come through here.
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
        """Ask the pump to look now rather than at the next tick."""
        try:
            self._wake.set()
        except Exception:
            pass

    async def _pump_loop(self) -> None:
        while True:
            try:
                self.pump_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                # A pump that dies stops recording every session at once, which is the
                # one failure this module exists to prevent. Log and keep going.
                logger.warning("bus: pump tick failed", exc_info=True)
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            self._wake.clear()

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
            return
        for _sid, worker in workers:
            try:
                worker.unattended_since = self._unattended_since
            except Exception:
                pass

    def pump_once(self) -> int:
        """Drain, journal and fan out one round. Returns how many events moved.

        Separate from the loop so a test can run exactly one tick, and so the shutdown
        path can flush what is still queued before the process goes.
        """
        self._publish_attachment()
        moved = 0
        for ev in self._collect():
            owner = ev.get("session_id") or ""
            extras = _pop_answer_extras(ev)
            if owner:
                try:
                    ev = self._log_for(owner).append_and_stamp(ev)
                except Exception:
                    logger.warning("bus: could not journal an event for %s", owner,
                                   exc_info=True)
            etype = ev.get("type")
            if owner:
                if etype in _CONCLUSIVE:
                    self._concluded[owner] = time.time()
                else:
                    # It is producing again, so whatever it concluded before is history.
                    self._concluded.pop(owner, None)
            if self._commit is not None and etype in _CONCLUSIVE:
                try:
                    self._commit(ev, extras)
                except Exception:
                    logger.warning("bus: turn commit failed for %s", owner, exc_info=True)
            for sub in list(self._subs):
                sub.offer(ev, extras)
            moved += 1
        return moved

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
            return events
        for _sid, worker in workers:
            try:
                events.extend(worker.drain())
            except Exception:
                logger.warning("bus: could not drain a worker", exc_info=True)
        return events


def _pop_answer_extras(ev: dict) -> dict:
    """Take the session-layer payload off an answer event.

    ``_full`` / ``_turn_start`` / ``_deferred`` are how a finished turn hands its
    history and its deferral record to whatever writes the session file. They are not
    part of the protocol and must reach neither the journal nor the wire.
    """
    if ev.get("type") != "answer":
        return {}
    return {k: ev.pop(k) for k in
            ("_full", "_turn_start", "_deferred", "_submitted_len", "_context_mode")
            if k in ev}

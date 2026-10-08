"""The event bus: one pump for the whole process, the journal as the record.

The journal is the bus and a socket is one of its followers. The pump lives for the
process, drains every worker whether or not anyone is listening, writes each event to
that session's ``transcript.jsonl`` under a monotonic ``seq``, and only then fans it out
to however many sockets happen to be subscribed — zero included:

    worker.out_q ──(pump)──► seq ──► transcript.jsonl      (durable, authoritative)
                               │
                               ├──► turn committer         (the session file)
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
from collections import deque
from typing import Callable, Iterable

from .job_wakes import DURABLE_EVENTS
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

# How long an attached socket has first refusal on an event before the pool acts on it
# regardless. Two consumers can act on a finished run or a finished turn — the socket's
# drain loop, which works against the conversation it holds on screen, and the pool,
# which works against the one on disk — and only one of them may: two turns for one run
# is a duplicate nothing downstream can undo, and two writers of one session file lose
# history silently.
#
# Presence of a subscription is not evidence that one will: a view whose drain loop has
# returned on a failed send, or that ended without closing its subscription, is a
# promise nobody is keeping, and the run it was holding then waits for somebody to open
# the panel and ask. So the right is granted for a bounded time rather than inferred
# from the subscriber list, and the pool takes over what nothing claimed. The drain
# loop reads its queue every 5 ms, so this only has to cover a tick plus one send; it
# is a latency bound on the handover and not a correctness condition, because whichever
# consumer claims first is the only one that acts.
_CLAIM_GRACE = 2.0

# How many claimed identities to remember. Only enough to answer the other consumer
# when it looks a tick later: past that nothing is asking, and an unbounded record of
# every run a long-lived server ever reported is a leak for no reader.
_CLAIMED_MEMORY = 512

# The event types that end a turn. An ``error`` is a conclusion as much as an ``answer``
# is — the turn is over either way — and conflating "concluded" with "succeeded" would
# leave a session that failed looking busy for ever.
_CONCLUSIVE = frozenset({"answer", "error"})

# Recorded, never delivered live. ``assistant_text`` is the aggregate of a streamed
# prose block, and a client that is attached has already had every one of its deltas:
# sending the aggregate as well would print the paragraph twice. It exists for the
# replay, where the deltas are gone — the journal keeps hundreds of `token` events out
# of itself, and without this aggregate a turn read back after an absence would have
# only its tool rows and its final answer, having lost everything the agent said in
# between.
_REPLAY_ONLY = frozenset({"assistant_text"})


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

    def __init__(self, pool, *, commit: Callable[[dict, dict], None] | None = None,
                 durable: Callable[[dict], None] | None = None,
                 interval: float = _PUMP_INTERVAL) -> None:
        self._pool = pool
        self._commit = commit
        # What takes in a finished run's wake when no socket will. The twin of
        # ``commit``, and for the same reason: a subscriber is a view, and a run that
        # ends with nobody looking still has to reach the turn it was launched from.
        self._durable = durable
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
        # Events nobody has claimed yet, by identity, each with what the pool would do
        # with it and the instant the sockets' first refusal expires:
        # ``{key: (event, extras, handler, deadline)}``. Emptied by every tick — either
        # a consumer claims the event or the sweep passes it to the handler — so it
        # holds at most one grace window's worth.
        self._unclaimed: dict[tuple, tuple[dict, dict, Callable, float]] = {}
        # The identities already taken, so "nobody ever offered this" and "somebody
        # has it" are different answers. Bounded and in order: a claim matters for as
        # long as it takes the other consumer to look, which is one tick.
        self._claimed: deque[tuple] = deque(maxlen=_CLAIMED_MEMORY)
        self.claimed_by_sweep = 0
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

    def attached(self) -> int:
        """How many clients are listening. Zero is the detached case."""
        return len(self._subs)

    # ── Who takes a durable event in ──────────────────────────────────────────

    @staticmethod
    def _claim_key(ev: dict) -> tuple:
        """The identity of a durable event, for deciding who acts on it.

        ``seq`` is in it because it is the one field guaranteed unique per event: a
        bulletin names no job, and the same finished run can be announced twice (by the
        watcher that was holding it and by a scan that found it unspoken-for), which are
        two events to be claimed separately and deduplicated downstream on the job key.
        """
        return (ev.get("session_id") or "", ev.get("type") or "",
                str(ev.get("job_key") or ""), ev.get("seq"))

    def claim(self, ev: dict) -> bool:
        """Take responsibility for *ev*. True for the first caller only.

        What a consumer calls before acting on a durable event. Both consumers ask, so
        neither has to know whether the other exists, and a run cannot get two turns
        however the timing falls.

        An event this bus never offered is granted: the pump is what offers them, so
        something that reached a consumer by another route reached only that one, and
        refusing it would lose a wake to protect against a second consumer that does
        not exist. Recorded either way, which is what makes the grant good for once.
        """
        key = self._claim_key(ev)
        if self._unclaimed.pop(key, None) is not None:
            self._claimed.append(key)
            return True
        if key in self._claimed:
            return False
        self._claimed.append(key)
        return True

    def unclaimed_for(self, session_id: str) -> bool:
        """Whether a conversation has an event still waiting to be claimed.

        What the pool asks before releasing an agent. An event in flight is work this
        process has accepted and not yet done, and the agent it is addressed to is the
        one that has to do it — so the window in which a socket is being given first
        refusal is not a window in which that agent may be closed.
        """
        if not session_id:
            return False
        return any(key[0] == session_id for key in self._unclaimed)

    def _offer(self, ev: dict, extras: dict, handler: Callable[[dict, dict], None]
               ) -> None:
        """Put *ev* up for claiming, and say how long the sockets have.

        No subscription means no first refusal to grant: the deadline is now, and the
        sweep at the end of this very tick calls *handler*. That is the detached case,
        and it costs it nothing — a run that ends with no window open is acted on inside
        the tick that drained it.
        """
        now = time.monotonic()
        self._unclaimed[self._claim_key(ev)] = (
            ev, extras, handler, now + (_CLAIM_GRACE if self._subs else 0.0))

    def _sweep_unclaimed(self, now: float | None = None) -> int:
        """Act on every event whose first-refusal window has closed. Returns how many.

        Run at the end of each tick. The event is claimed here before its handler runs,
        so a drain loop that wakes up a moment later finds it taken and does not do the
        same work a second time.
        """
        if not self._unclaimed:
            return 0
        at = time.monotonic() if now is None else now
        due = [key for key, entry in self._unclaimed.items() if entry[3] <= at]
        taken = 0
        for key in due:
            entry = self._unclaimed.pop(key, None)
            if entry is None:
                continue
            ev, extras, handler, _deadline = entry
            self._claimed.append(key)
            self.claimed_by_sweep += 1
            taken += 1
            if self._subs:
                # Granted and not taken up. Said out loud because it is the only
                # outward sign that a socket is subscribed and doing nothing with what
                # it is sent.
                logger.warning("bus: no attached view claimed %s for %s within %.0fs; "
                               "handling it here", ev.get("type"),
                               ev.get("session_id"), _CLAIM_GRACE)
            try:
                handler(ev, extras)
            except Exception:
                logger.warning("bus: handling %s for %s failed", ev.get("type"),
                               ev.get("session_id"), exc_info=True)
        return taken

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
        if self._unclaimed or self.claimed_by_sweep:
            rows.append({"label": "claims", "detail":
                         f"{len(self._unclaimed)} awaiting a claim, "
                         f"{self.claimed_by_sweep} handled by the pool unclaimed"})
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
        self.ticks += 1
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
            # After the journal, so the event every consumer reads is the stamped one,
            # and offered rather than acted on: an attached socket gets first refusal
            # (it works against the conversation it holds on screen), the sweep below
            # does what nobody claimed. Two turns for one finished run, or two writers
            # of one session file, are duplicates nothing downstream can undo — and the
            # claim, not the subscriber list, is what elects the one that acts.
            if self._commit is not None and etype in _CONCLUSIVE:
                self._offer(ev, extras, self._commit)
            if self._durable is not None and etype in DURABLE_EVENTS:
                self._offer(ev, {}, lambda e, _x: self._durable(e))
            if isinstance(ev.get("seq"), int):
                self.journaled += 1
            if etype in _REPLAY_ONLY:
                continue   # journaled above; the live client had the deltas
            for sub in list(self._subs):
                sub.offer(ev, extras)
                self.fanned_out += 1
            moved += 1
        # Last, so an event offered in this tick with no socket to claim it is taken in
        # within it: with nothing attached, the whole handover is one synchronous tick.
        self._sweep_unclaimed()
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
    history and its deferral record to whatever writes the session file, and
    ``_unconsumed_steer`` the steering it ended without reading. They are not
    part of the protocol and must reach neither the journal nor the wire.
    """
    if ev.get("type") != "answer":
        return {}
    return {k: ev.pop(k) for k in
            ("_full", "_turn_start", "_deferred", "_submitted_len", "_context_mode",
             "_unconsumed_steer")
            if k in ev}

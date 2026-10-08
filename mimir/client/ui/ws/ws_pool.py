"""One agent per conversation, created when needed and released when idle.

One worker per session, each with its own agent, thread and serial query loop. That is
what lets a turn keep running in a conversation nobody is looking at: a worker shared
between conversations can stream only one of them anywhere the user can see, which forces
leaving a conversation to stop its turn.

Three things make it affordable:

**Lazily.** A worker is built on its session's first query, not when the session is
created. Construction blocks on the LLM backend and then spawns ~19 MCP servers, which on
a cold vLLM is minutes — so that cost falls only on a conversation that actually asks
something, is paid off the event loop, and is announced to the session paying it.

**Capped.** ``MIMIR_MAX_LIVE_SESSIONS`` live workers, three by default. The binding
constraint is processes, not tokens: three agents is some sixty Python interpreters. Past
the cap a turn waits for a slot and the session is told so — a queue nobody can see reads
as a hang.

**Released.** An idle worker is closed and its servers with it. Never one that is busy,
parked on a card, watching a background job, or on screen: each of those is work or
attention that evicting would throw away.

The pool is process-global, not per connection: two webviews on one port must see the
same live turns.

Every mutation happens on the WS event loop, so an ``asyncio.Lock`` is the whole
concurrency story here; nothing touches ``_workers`` from a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from typing import Any, Callable, Iterator

from . import server_registry
from .event_bus import _EventBus
from .job_scan import has_baseline, mark_wakes_reported, scan_all_sessions
from .job_wakes import DURABLE_EVENTS, checkin_text, wake_text
from .session_store import SessionStore
from .turn_commit import commit_answer
from .ws_worker import _AgentWorker, _detach_grace

logger = logging.getLogger(__name__)

# How many agents may be alive at once. The ceiling is the process count: each agent
# spawns its own set of MCP servers, so this multiplies by ~19. Three is a working
# default for a workstation; lower it on a shared login node.
_DEFAULT_CAP = 3

# How long a worker may sit unused before it is released. Long enough that switching back
# and forth does not pay for a rebuild, short enough that a forgotten conversation stops
# holding twenty subprocesses.
_DEFAULT_IDLE_TTL = 600.0

# How long everything must stay quiet before a detached server stops on its own. Long
# enough that closing a laptop lid for a meeting does not end a run, short enough that a
# forgotten workspace does not hold sixty interpreters overnight.
_DEFAULT_SERVER_IDLE_TTL = 7200.0


def _server_idle_ttl() -> float:
    try:
        return max(60.0, float(
            os.environ.get("MIMIR_SERVER_IDLE_TTL", "") or _DEFAULT_SERVER_IDLE_TTL))
    except ValueError:
        return _DEFAULT_SERVER_IDLE_TTL


# How often the reaper looks. Nothing here is urgent — a worker that lives thirty seconds
# past its welcome costs nothing — and a tight tick would walk the pool for no reason.
_REAP_INTERVAL = 30.0


def _cap() -> int:
    try:
        value = int(os.environ.get("MIMIR_MAX_LIVE_SESSIONS", "") or _DEFAULT_CAP)
    except ValueError:
        return _DEFAULT_CAP
    return max(1, value)


def _idle_ttl() -> float:
    try:
        return max(30.0, float(os.environ.get("MIMIR_SESSION_IDLE_TTL", "") or _DEFAULT_IDLE_TTL))
    except ValueError:
        return _DEFAULT_IDLE_TTL


class _Settings:
    """The UI knobs a newly built worker must start from.

    A worker is built long after the user chooses these, so it has to arrive already
    configured: without the record, a conversation started after the user picked "plan
    mode" would quietly come up in agent mode. Each setter therefore writes its argument
    here *and* applies it to every live worker, and a new worker replays the record once
    its agent exists.

    Deliberately a recording of calls rather than a dict of values: the worker's setters
    are where the validation lives, and duplicating "what a mode may be" here is how the
    two drift apart.
    """

    def __init__(self) -> None:
        self._calls: dict[str, tuple] = {}

    def record(self, name: str, *args: Any) -> None:
        self._calls[name] = args

    def forget(self, name: str) -> None:
        self._calls.pop(name, None)

    def replay(self, worker: _AgentWorker) -> None:
        for name, args in self._calls.items():
            method = getattr(worker, name, None)
            if method is None:
                continue
            try:
                method(*args)
            except Exception:
                logger.warning("pool: could not apply %s to a new worker", name,
                               exc_info=True)

    def recorded(self, name: str) -> tuple | None:
        return self._calls.get(name)


class _AgentPool:
    """The live workers, keyed by the session each one works for."""

    def __init__(self, model: str, *, cap: int | None = None,
                 idle_ttl: float | None = None) -> None:
        self.model = model
        self.cap = cap if cap is not None else _cap()
        self.idle_ttl = idle_ttl if idle_ttl is not None else _idle_ttl()
        # The session the front-end is showing — a property of the view, not of any one
        # worker, which is why it lives on the pool.
        self.active_session_id: str | None = None
        self.settings = _Settings()

        # The stand-in for a conversation with no agent yet, which is the ordinary state
        # until its first query. One per pool, shared: it holds no session's state because
        # it holds no state at all beyond the model.
        self._detached = _AgentWorker.detached(model)

        self._workers: dict[str, _AgentWorker] = {}
        self._last_use: dict[str, float] = {}
        # Sessions whose worker is being built right now, so two queries arriving
        # together do not build two agents for one conversation.
        self._building: dict[str, asyncio.Future] = {}
        # Submissions waiting for a slot: (session_id, callable taking the worker).
        self._queue: deque[tuple[str, Callable[[_AgentWorker], None]]] = deque()
        self._lock = asyncio.Lock()
        self._reaper: asyncio.Task | None = None

        # The one pump for this process. It drains every worker, journals what they
        # emit and fans it out to whatever sockets are attached — zero included, which
        # is the point: the pool outlives every connection, so what records its output
        # has to as well.
        self.store = SessionStore()
        self.bus = _EventBus(self, commit=self._commit_turn,
                             durable=self.consume_durable_event)

        # Resolved when this process decides it is no longer needed. Awaited beside the
        # signal handlers, so an idle stop and a SIGTERM take the same path out — the
        # one that closes the MCP servers.
        self.stop_requested: asyncio.Event | None = None
        self.server_idle_ttl = _server_idle_ttl()
        # When the whole process first looked idle, or None while it does not. The TTL
        # is measured from here, so a moment of activity restarts it rather than
        # shortening it.
        self._idle_since: float | None = None
        # Finished runs whose wake no turn has taken in yet, by session, each entry
        # remembering whether the user has already been shown it. The pool's own copy
        # of what a connection keeps, so a wake steered into a running turn is still
        # owed a turn of its own if that turn never read it.
        self._wakes_pending: dict[str, list[dict]] = {}
        # The last bulletin a busy conversation was not interrupted with, by session.
        self._held_checkin: dict[str, dict] = {}

    # ── Is this process still needed ──────────────────────────────────────────

    def idle_report(self) -> dict:
        """Why this process is or is not still needed, in one interrogable answer.

        **The criterion is positive.** "Not busy" is not "has finished": a worker is
        also not busy between queries, after a turn that broke, and while it waits
        behind a card. An idle test built on the absence of noise would stop a server
        whose turn merely paused, so a conversation counts as finished only when it has
        actually *concluded* — delivered a final answer (or ended in an error, which is
        also an ending) — which the bus records as it hands that event to the committer.

        And a run launched but not collected is work in progress even with no turn
        running: that is precisely the "I submitted a two-hour build and left" case
        this whole path exists for, so a live job on disk holds the process open — and
        so does one that has *ended* without its wake reaching a turn, which is the same
        debt one step later.

        The rest are not activity but prohibitions: somebody is attached, a card is
        parked, a deferral is owed, a turn is queued for a slot.

        One place, deliberately: the idle shutdown reads it, and so does the question
        "has this window anything worth leaving running?".
        """
        reasons: list[str] = []
        attached = self.bus.attached()
        if attached:
            reasons.append(f"{attached} client(s) attached")
        if self._queue:
            reasons.append(f"{len(self._queue)} turn(s) waiting for a slot")
        if self._building:
            reasons.append(f"{len(self._building)} agent(s) being built")

        unfinished: list[str] = []
        for session_id, worker in list(self._workers.items()):
            try:
                if worker.has_work_pending():
                    reasons.append(f"session {session_id} has a turn in flight")
                    continue
            except Exception:
                reasons.append(f"session {session_id} could not be asked")
                continue
            if getattr(worker, "_pending_prompt", None) is not None:
                reasons.append(f"session {session_id} is parked on a card")
                continue
            if getattr(worker, "has_deferral", False):
                reasons.append(f"session {session_id} is owed an answer")
                continue
            if self.bus.concluded_at(session_id) is None:
                # Never answered, and not running: a turn that ended without saying so.
                # Counted as working, because nothing observed it finishing.
                unfinished.append(session_id)
        if unfinished:
            reasons.append("session(s) that never delivered an answer: "
                           + ", ".join(sorted(unfinished)))

        live_jobs, owed_wakes = self._job_keys()
        if live_jobs:
            reasons.append("background run(s) still going: " + ", ".join(live_jobs))
        if owed_wakes:
            reasons.append("run(s) whose wake nobody has taken in yet: "
                           + ", ".join(owed_wakes))

        return {"idle": not reasons, "reasons": reasons}

    @staticmethod
    def _job_keys() -> tuple[list[str], list[str]]:
        """The runs still going, and the finished ones still owed a wake.

        Read from disk rather than from the live workers: a run outlives the agent that
        launched it, and the session it belongs to may have no worker at all right now.
        One scan answers both, since ``scan_all_sessions`` already drops the runs whose
        wake has been delivered — what it returns is exactly "still going" plus "ended
        and unspoken for".

        **Why an owed wake holds the process open.** A finished run that nothing has
        taken in is a turn this process still has to start, which is work in progress
        as surely as the run itself was. Stopping here is how a conversation loses the
        night: the job ends, its wake is journaled, and the server that owes the turn
        shuts down two hours later without ever having started it.

        Only sessions that have been scanned before are counted. A job directory is
        never swept, however old, so an unbaselined session holds every run it ever
        finished with no marker on any of them — reading that history as debt would
        leave a workspace permanently un-stoppable, which is worse than stopping one
        with a wake outstanding and exactly the trade the baseline exists to make.
        """
        live: list[str] = []
        owed: list[str] = []
        try:
            for session_id, jobs in scan_all_sessions().items():
                baselined = has_baseline(session_id)
                for job in jobs:
                    if job.live:
                        live.append(f"{session_id}/{job.job_key}")
                    elif baselined:
                        owed.append(f"{session_id}/{job.job_key}")
        except Exception:
            # Unreadable means unknown, and unknown must not read as "nothing is
            # running": erring the other way stops a server mid-build.
            logger.warning("pool: the detached runs could not be read", exc_info=True)
            return ["<unreadable>"], []
        return sorted(live), sorted(owed)

    def request_stop(self) -> None:
        """Ask the serve loop to shut down. Idempotent."""
        event = self.stop_requested
        if event is not None and not event.is_set():
            event.set()

    def _commit_turn(self, ev: dict, extras: dict) -> None:
        """Write a finished turn into its session file, and carry on from there.

        Reached only for an answer no attached view claimed. A connected ``_Session``
        claims and writes its own — to ``self.history`` for the conversation on screen,
        through ``_persist_detached_answer`` for any other — and two live writers of one
        file lose history silently, so the claim is what elects exactly one of us. This
        covers the case a connection cannot: nobody looking, and a turn that would
        otherwise run, cost its tokens and vanish.

        Writing it down is half of what a connection does when an answer lands. The
        other half is carrying on — see :meth:`_carry_on_after_turn` — and a detached
        process that did only the first half answers the step that finished and stops
        there.
        """
        if ev.get("type") != "answer":
            return
        session_id = ev.get("session_id")
        if not session_id:
            return
        if not ev.get("cancelled"):
            result = commit_answer(
                self.store, session_id, ev, extras,
                submitted_len=extras.get("_submitted_len"),
                context_mode=extras.get("_context_mode") or "full",
            )
            if result is not None:
                logger.info("commit: session %s wrote its answer with nobody attached",
                            session_id)
        # After the write, so a turn started here is handed the history this answer
        # just produced rather than the one it inherited.
        try:
            self._carry_on_after_turn(session_id, extras)
        except Exception:
            logger.warning("session %s could not be carried on after its turn",
                           session_id, exc_info=True)

    def consume_durable_event(self, ev: dict) -> None:
        """Turn a finished run's wake into a turn when no socket will.

        Every way this can fail is a logged return rather than a raise, and the pump
        guards the call besides: one unreadable session must not stop the tick that is
        draining every other conversation's output.

        The counterpart of :meth:`_commit_turn`. A watcher outlives every connection —
        it is a task on a worker's loop, and the pool refuses to release a worker that
        holds one — so a two-hour run reports in whether or not a window is open, and
        the pump journals what it says. Acting on that report is this method's job, and
        it must not need a socket either: a wake only a connection can route is a
        conversation that waits out the night for a job that finished in three minutes,
        and resumes when somebody opens the panel and asks.

        Called by the pump, which has already checked that nobody is attached: a
        ``_Session`` present routes this itself, against the history it holds on screen.

        Everything here is synchronous on purpose. Loading a session, appending to it,
        saving it and queueing a turn are all plain calls — ``submit_query`` is a queue
        put — so this needs no loop of its own and runs inside the pump tick that
        drained the event.

        The turn's answer is written back by :meth:`_commit_turn`, off the same pump,
        against the ``_submitted_len`` the worker records as it takes the turn in. So a
        wake taken in here is a complete turn: asked, answered and persisted, with no
        connection involved at any point.
        """
        etype = ev.get("type")
        owner = ev.get("session_id")
        if not owner or etype not in DURABLE_EVENTS:
            return
        checkin = etype == "job_checkin"
        worker = self.get(owner)
        if worker is None:
            # Not reachable by any emitter that goes through the bus, and said out loud
            # rather than handled: every one of them runs on a worker's own loop — a
            # watcher, the check-in cycle, the scan a build does as it comes up — and
            # ``releasable`` refuses to close a worker while a job of its is watched, a
            # wake of its is owed, its output is undrained or an event of its is still
            # to be claimed. So this is a broken invariant, not a case.
            #
            # The run is left unsettled on disk, which is the honest state: nothing has
            # taken its wake in, and the next agent built for this conversation
            # announces it as it comes up.
            logger.warning("%s for job %r arrived for session %s, whose agent is "
                           "closed — the run stays owed", etype, ev.get("job_key"),
                           owner)
            return
        if checkin:
            if self.is_busy(owner) or self.is_parked(owner):
                # A bulletin interrupts nothing, by design: steering "nothing to
                # report" into a turn makes the agent answer about the job instead of
                # the work. Held, not dropped, and delivered when that turn lands — the
                # twin of what a connection does with one.
                self._held_checkin[owner] = ev
                return
            self._submit_checkin(owner, worker, ev)
            return
        pending = self._wakes_pending.setdefault(owner, [])
        # One wake per run, however many times it is announced. Two emitters can speak
        # before either delivers — the watcher that was holding the run, and a disk
        # scan that found it unspoken-for — and the marker that settles it is written
        # at delivery, so the dedup has to live here.
        if any(item["ev"].get("job_key") == ev.get("job_key") for item in pending):
            logger.info("wake for job %r of session %s already pending; dropping the "
                        "duplicate announcement", ev.get("job_key"), owner)
            return
        item = {"ev": ev, "told": False}
        pending.append(item)
        if self.is_busy(owner) or self.is_parked(owner):
            # Handed to the turn already running, which learns at its next step
            # boundary that the run finished and carries on — no extra turn, no second
            # final answer. Left pending deliberately: a steer is only known to have
            # been read when the loop says so, and the answer that ends that turn says
            # which ones it never read.
            self._record_wake(owner, [ev])
            item["told"] = True
            worker.submit_steer(wake_text(ev))
            return
        self._flush_wakes(owner, worker)

    # ── The twin of what a connection does with a wake ────────────────────────

    def _record_wake(self, owner: str, events: list[dict]) -> None:
        """Write the bubbles the user finds on their return, and journal the wakes.

        Deliberately not the history: which message a turn is *given* is the flush's
        business, and a wake steered into a running turn arrives in that turn's own
        messages instead. The journal entry is what a window opening later replays, and
        what the mechanism is audited from.
        """
        if not events:
            return
        try:
            session = self.store.load_session(owner)
        except Exception:
            logger.warning("the wake of session %s could not be written down: it "
                           "could not be loaded", owner, exc_info=True)
            return
        for ev in events:
            session.display_messages.append(
                {"role": "system", "kind": "text", "text": f"🔔 {wake_text(ev)}"})
        try:
            self.store.save_session(session)
        except Exception:
            logger.warning("the wake of session %s could not be written down: it "
                           "would not save", owner, exc_info=True)
            return
        for ev in events:
            self.bus.record_client_event(owner, {
                "type": "job_wake", "text": wake_text(ev), "job": ev.get("job_key")})

    def _flush_wakes(self, owner: str, worker: _AgentWorker) -> bool:
        """Start one turn carrying every wake this conversation is still owed.

        One turn for all of them, not one apiece: a burst that finished together is one
        piece of news, and three turns would answer the first and then re-answer it
        twice. Every pending wake goes in, including ones already steered — a steer is
        only known to have been read when the loop says so, and re-telling a run is
        recoverable where losing one is not.
        """
        items = self._wakes_pending.pop(owner, [])
        if not items:
            return False
        events = [item["ev"] for item in items]
        fresh = [item["ev"] for item in items if not item["told"]]
        text = "\n\n".join(wake_text(ev) for ev in events)
        self._record_wake(owner, fresh)
        if not self._submit_turn(owner, worker, text):
            # Nothing was started, so nothing may be settled: put them back and let
            # the next announcement or the next scan try again.
            self._wakes_pending[owner] = items + self._wakes_pending.get(owner, [])
            return False
        mark_wakes_reported(events)
        logger.info("session %s woken by %s with nobody attached",
                    owner, ", ".join(str(ev.get("job_key")) for ev in events))
        return True

    def _submit_checkin(self, owner: str, worker: _AgentWorker, ev: dict) -> None:
        """Ask this conversation for one line about the runs it is still waiting on.

        No bubble: the instruction is addressed to the model, and a conversation checked
        on twenty times would reopen on twenty blocks of it, every one above the answer
        it had produced. The journal still records it, which is what it is audited from.
        """
        self._held_checkin.pop(owner, None)
        text = checkin_text(ev)
        if self._submit_turn(owner, worker, text):
            self.bus.record_client_event(
                owner, {"type": "job_checkin", "text": text,
                        "job": ev.get("job_key")})

    def _submit_turn(self, owner: str, worker: _AgentWorker, text: str) -> bool:
        """Append *text* to the stored conversation and queue a turn on its own agent.

        The turn's answer is written back by :meth:`_commit_turn`, off the same pump,
        against the ``_submitted_len`` the worker records as it takes the turn in. So a
        turn started here is a complete one: asked, answered and persisted, with no
        connection involved at any point.
        """
        try:
            session = self.store.load_session(owner)
        except Exception:
            logger.warning("no turn started for session %s: it could not be loaded",
                           owner, exc_info=True)
            return False
        message = {"role": "user", "content": text}
        session.llm_history.append(message)
        session.llm_history_full.append(dict(message))
        try:
            self.store.save_session(session)
        except Exception:
            logger.warning("no turn started for session %s: it would not save", owner,
                           exc_info=True)
            return False
        worker.submit_query(text, list(session.llm_history), session_id=owner)
        return True

    def _carry_on_after_turn(self, owner: str, extras: dict) -> None:
        """Do what a connection does the moment a turn lands, with none attached.

        A turn ending is not the end of the work, and this is the difference between a
        chain of steps that runs overnight and one that stops at its first link. A
        socket, on every answer, puts the steering the loop never read to a new turn,
        starts a turn for every run that finished while it was busy, and delivers the
        bulletin it was holding. None of that is in the committer — a commit that
        insisted on a connection could not run without one — so it is here.

        Which steers the loop *did* read is on the answer itself: ``_unconsumed_steer``
        is what it ended without taking in. A wake named there was never read and is
        still owed a turn; one not named there was read inside the turn that just
        answered, and is settled here, which is the only moment that is knowable.
        """
        items = self._wakes_pending.pop(owner, [])
        unread = {text for text in (extras.get("_unconsumed_steer") or []) if text}
        kept, read = [], []
        for item in items:
            if item["told"] and wake_text(item["ev"]) not in unread:
                read.append(item["ev"])
            else:
                kept.append(item)
        if read:
            # The turn that just answered took these in and answered for them. Settling
            # them here is what keeps the next scan from re-announcing a run the model
            # has already dealt with.
            mark_wakes_reported(read)
        worker = self.get(owner)
        if kept:
            self._wakes_pending[owner] = kept + self._wakes_pending.get(owner, [])
            # An owed wake is one of the things that keeps this agent from being
            # released, so there is one here; the guard is for the broken invariant,
            # not for a case.
            if worker is not None and not (self.is_busy(owner) or self.is_parked(owner)):
                self._flush_wakes(owner, worker)
            return
        held = self._held_checkin.get(owner)
        if held is None or worker is None:
            return
        # Dropped rather than delivered when the runs it described have since finished:
        # their completion wakes say everything it would, and better.
        try:
            still_running = bool(worker.watched_job_keys())
        except Exception:
            still_running = False
        if not still_running:
            self._held_checkin.pop(owner, None)
            return
        if not (self.is_busy(owner) or self.is_parked(owner)):
            self._submit_checkin(owner, worker, held)

    # ── Looking up ────────────────────────────────────────────────────────────

    def get(self, session_id: str | None) -> _AgentWorker | None:
        """The live worker for *session_id*, or None. Never builds one."""
        if not session_id:
            return None
        worker = self._workers.get(session_id)
        if worker is not None:
            self._last_use[session_id] = time.monotonic()
        return worker

    def worker_or_detached(self, session_id: str | None) -> _AgentWorker:
        """The live worker for *session_id*, or the detached stand-in.

        What the session reads for everything that is not a query: the settings behind
        the greeting, a toggle panel, a mode switch typed before anything was asked. Each
        of those has a right answer with no agent — the worker's own getters give it — so a
        stand-in lets every call site that addresses "the conversation on screen" read it
        directly, without a None check of its own.
        """
        return self.get(session_id) or self._detached

    def items(self) -> Iterator[tuple[str, _AgentWorker]]:
        """Every live worker, as (session_id, worker). A snapshot: the caller may drain."""
        return iter(list(self._workers.items()))

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._workers

    def __len__(self) -> int:
        return len(self._workers)

    def queued_position(self, session_id: str) -> int | None:
        """Where *session_id* sits in the waiting line, 1-based, or None."""
        for index, (sid, _submit) in enumerate(self._queue, start=1):
            if sid == session_id:
                return index
        return None

    # ── Building ──────────────────────────────────────────────────────────────

    async def worker_for(self, session_id: str, *,
                         on_wait: Callable[[], Any] | None = None) -> _AgentWorker | None:
        """The worker for *session_id*, building one if there is room.

        None means "no slot right now" — the caller queues its submission with
        :meth:`enqueue` rather than running it. *on_wait* is called once if a build is
        actually starting, so the session can say why it is about to be quiet for a
        while; it is not called for a worker that already exists.
        """
        existing = self.get(session_id)
        if existing is not None:
            return existing

        async with self._lock:
            existing = self._workers.get(session_id)
            if existing is not None:
                self._last_use[session_id] = time.monotonic()
                return existing
            pending = self._building.get(session_id)
            if pending is None and len(self._workers) >= self.cap:
                if not await self._release_one_idle_locked():
                    return None
            if pending is None:
                pending = asyncio.get_running_loop().create_future()
                self._building[session_id] = pending
                starting = True
            else:
                starting = False

        if not starting:
            # Someone else is building it; wait for their result rather than racing.
            return await asyncio.shield(pending)

        if on_wait is not None:
            try:
                on_wait()
            except Exception:
                pass
        try:
            worker = await self._build(session_id)
        except BaseException as exc:
            async with self._lock:
                self._building.pop(session_id, None)
            if not pending.done():
                pending.set_exception(exc)
            # Nobody may be awaiting the future; retrieving the exception here keeps
            # asyncio from reporting it a second time as never retrieved.
            pending.exception()
            raise
        async with self._lock:
            self._building.pop(session_id, None)
            self._workers[session_id] = worker
            self._last_use[session_id] = time.monotonic()
        if not pending.done():
            pending.set_result(worker)
        self.ensure_reaper()
        self.ensure_pump()
        return worker

    async def _build(self, session_id: str) -> _AgentWorker:
        """Construct a worker off the event loop.

        ``_AgentWorker.__init__`` blocks until its agent is up — the LLM backend first,
        then ~19 MCP servers — so building it inline would stop the drain loops of every
        other conversation for the duration.
        """
        loop = asyncio.get_running_loop()
        worker = await loop.run_in_executor(
            None, lambda: _AgentWorker(self.model, session_id=session_id))
        worker.active_session_id = self.active_session_id
        self.settings.replay(worker)
        # What the last agent of this conversation had learned and set aside. A panel
        # reopening a conversation restores this; a rebuild driven by anything else —
        # a finished run that needs an agent to be answered for, a queued turn admitted
        # after a release — is the same agent coming back and must start from the same
        # place. Here rather than at the call sites, because this is the one place a
        # worker is constructed for a session.
        try:
            stored = self.store.load_session(session_id)
        except Exception:
            stored = None
        if stored is not None and getattr(stored, "carry_context", None):
            try:
                worker.load_agent_state({"carry_context": stored.carry_context})
            except Exception:
                logger.warning("pool: session %s could not resume its carried context",
                               session_id, exc_info=True)
        # On the loop, not in the executor: putting a watcher back creates a task, and
        # a task created off a running loop never polls anything. Here rather than in
        # the worker's constructor for the same reason — the constructor runs in the
        # executor, where there is no loop to host a watcher.
        #
        # A run this session left behind is un-watched by construction here: the worker
        # is new, so nothing of its is holding anything. That makes this the one place
        # where "the promise to report a run" can be re-made without being asked.
        try:
            worker.rearm_detached_jobs()
        except Exception:
            logger.warning("pool: could not re-arm the detached runs of session %s",
                           session_id, exc_info=True)
        return worker

    # ── Queueing past the cap ─────────────────────────────────────────────────

    def enqueue(self, session_id: str, submit: Callable[[_AgentWorker], None]) -> int:
        """Hold *submit* until a slot frees. Returns its 1-based place in the line."""
        self._queue.append((session_id, submit))
        self.ensure_reaper()
        self.ensure_pump()
        return len(self._queue)

    def drop_queued(self, session_id: str) -> None:
        """Forget anything *session_id* was waiting to run (it was deleted, or reset)."""
        self._queue = deque(
            entry for entry in self._queue if entry[0] != session_id)

    async def pump(self) -> list[str]:
        """Admit as many queued submissions as there are slots. Returns those admitted."""
        admitted: list[str] = []
        while self._queue:
            session_id, submit = self._queue[0]
            worker = await self.worker_for(session_id)
            if worker is None:
                break
            self._queue.popleft()
            try:
                submit(worker)
            except Exception:
                logger.warning("pool: queued submission for %s failed", session_id,
                               exc_info=True)
            admitted.append(session_id)
        return admitted

    # ── Releasing ─────────────────────────────────────────────────────────────

    def releasable(self, session_id: str) -> bool:
        """Whether this worker may be closed without throwing work or attention away.

        Each clause is a thing that would be lost:

        * **busy** — a turn is running, or is queued and about to.
        * **watching a background job** — a two-hour build's watcher lives on this
          worker's loop. Evicting it loses the wake that watcher exists to deliver, so
          the build finishes and nothing ever says so.
        * **parked on a card** — the turn is waiting on a person, with no timeout, by
          design. Closing it discards a question the user may be about to answer.
        * **holding a deferral** — the same debt, set aside. A deferred turn has
          cleared its pending card, so the clause above no longer sees it, and the
          user still owes it an answer; releasing the agent here throws away the ~19
          servers that answer will be resumed against.
        * **owed a wake** — a run of its finished and no turn has taken that news in
          yet. The debt one step later than "watching a job", and the clause that
          closes the gap between them: a watcher drops the job the instant it reports
          it, so without this the agent is releasable for the one moment at which it
          is needed most — and the conversation it belongs to has by definition been
          idle for hours, because it was waiting on a long run.
        * **holding undrained output, or an event still to be claimed** — its last
          word has not been acted on. The pump empties ``out_q`` every 50 ms and, with
          nothing attached, does the whole handover inside that same synchronous tick;
          an attached socket is given a moment's first refusal first. Across either,
          closing the worker discards what it had just said — the completion event
          included — or closes the very agent that event is addressed to.
        * **on screen** — the conversation the user is reading must stay instant.

    Together these are what make "ten minutes idle" safe to act on. Idleness is counted
    from the last time the pool was asked for the worker, which for a conversation
    waiting on an overnight run is hours ago throughout — so the clock never protects
    it and these clauses are the whole of what does. Their sum is one invariant: an
    agent is released only when nothing of its is unresolved, which is why a wake can
    never arrive for a conversation whose agent has been closed.
        """
        worker = self._workers.get(session_id)
        if worker is None:
            return False
        if session_id == self.active_session_id:
            return False
        try:
            if worker.has_work_pending():
                return False
        except Exception:
            return False
        if getattr(worker, "_bg_jobs", None):
            return False
        if getattr(worker, "_pending_prompt", None) is not None:
            return False
        if getattr(worker, "has_deferral", False):
            return False
        if self._wakes_pending.get(session_id):
            return False
        bus = getattr(self, "bus", None)
        if bus is not None:
            try:
                if bus.unclaimed_for(session_id):
                    return False
            except Exception:
                return False
        out_q = getattr(worker, "out_q", None)
        if out_q is not None:
            try:
                if not out_q.empty():
                    return False
            except Exception:
                return False
        return True

    def idle_for(self, session_id: str) -> float:
        return time.monotonic() - self._last_use.get(session_id, 0.0)

    async def _release_one_idle_locked(self) -> bool:
        """Free a slot by closing the longest-idle releasable worker. Caller holds the lock.

        Returns False when nothing may be released — every worker is working, waiting on
        the user, or being read. The caller then queues instead of evicting: taking a slot
        from a conversation that is mid-task to give it to a new one trades a visible wait
        for silent lost work.
        """
        candidates = sorted(
            (sid for sid in self._workers if self.releasable(sid)),
            key=self.idle_for, reverse=True,
        )
        if not candidates:
            return False
        await self._close_locked(candidates[0], reason="a slot was needed")
        return True

    async def release_idle(self) -> list[str]:
        """Close every releasable worker idle past the TTL. Returns the sessions closed."""
        closed: list[str] = []
        async with self._lock:
            for session_id in list(self._workers):
                if not self.releasable(session_id):
                    continue
                if self.idle_for(session_id) < self.idle_ttl:
                    continue
                await self._close_locked(session_id, reason="idle")
                closed.append(session_id)
        return closed

    async def close(self, session_id: str) -> None:
        """Close one worker outright — its conversation is gone."""
        async with self._lock:
            await self._close_locked(session_id, reason="its session was deleted")
        self.drop_queued(session_id)

    async def _close_locked(self, session_id: str, *, reason: str) -> None:
        worker = self._workers.pop(session_id, None)
        self._last_use.pop(session_id, None)
        if worker is None:
            return
        logger.info("pool: releasing the agent of session %s (%s)", session_id, reason)
        # Off the loop: aclose waits for the MCP servers to go, which is seconds of
        # subprocess teardown, and the event loop is serving every other conversation.
        await asyncio.get_running_loop().run_in_executor(None, worker.aclose)

    async def aclose_all(self) -> None:
        """Close every agent, all at once.

        Together rather than one after another: each close waits on its own MCP servers
        going away, so in series the whole shutdown costs that wait times the number of
        live conversations — a minute of a process that has already been asked to stop,
        while its replacement is starting beside it. They share nothing, so there is
        nothing for the serialisation to protect.
        """
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        # Flush before stopping: whatever a turn emitted in its last moments is still
        # on ``out_q``, and the journal is the only thing that will remember it.
        try:
            self.bus.pump_once()
        except Exception:
            logger.warning("pool: final pump flush failed", exc_info=True)
        await self.bus.aclose()
        async with self._lock:
            workers = list(self._workers.items())
            self._workers.clear()
            self._last_use.clear()
        if not workers:
            return
        loop = asyncio.get_running_loop()
        for session_id, _worker in workers:
            logger.info("pool: releasing the agent of session %s (shutting down)",
                        session_id)
        await asyncio.gather(
            *(loop.run_in_executor(None, worker.aclose) for _sid, worker in workers),
            return_exceptions=True,
        )

    # ── The reaper ────────────────────────────────────────────────────────────

    def ensure_pump(self) -> None:
        """Start the event pump, if it is not already running.

        Idempotent and loop-dependent in the same way as :meth:`ensure_reaper`, and
        called from the same places, so a worker built on a bare pool in a test does
        not need a running loop.
        """
        try:
            self.bus.start()
        except RuntimeError:
            pass   # no loop (tests driving the pool synchronously call pump_once)

    def ensure_reaper(self) -> None:
        """Start the idle sweep, if it is not already running."""
        if self._reaper is not None and not self._reaper.done():
            return
        try:
            self._reaper = asyncio.get_running_loop().create_task(self._reap_loop())
        except RuntimeError:
            self._reaper = None   # no loop (tests calling the pool synchronously)

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL)
            try:
                await self.release_idle()
                await self.pump()
                self._consider_stopping()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("pool: idle sweep failed", exc_info=True)

    def _nobody_asked_to_keep_it(self) -> bool:
        """Whether this process has lost its last window with no claim on it.

        A server exists in one of two states and there is no third: claimed by at least
        one conversation — detached, which redirected its output to a log and asked not
        to be killed — or owned by the window that started it, dying with it. What used
        to be possible was a server in neither: one a window had merely attached to,
        nobody's to kill, still writing to a pipe whose reader is gone. It kept working,
        indifferently and unobserved, and nobody had asked it to.

        So a server nobody claimed stops when its last client has been gone for the
        grace. Server-side, because the only place that can decide this reliably is the
        process itself: an extension host that is dying has no time to end a process it
        does not own, and may be killed before it tries. The grace is the one a parked
        card already uses — a window reload closes and reopens the socket, and that is
        not a departure.

        This is deliberately not conditioned on what is in flight. A run still going
        does not earn a server the right to survive unasked: the panel asks before the
        window goes, and "keep them going" is what claims it. The run itself is not
        killed either way — it has its own process session and an exit code trap, and
        the next agent built for its conversation picks it up again.
        """
        if self.bus is None or not self.bus.ever_attached:
            return False
        unattended = self.bus.unattended_for()
        if unattended is None or unattended < _detach_grace():
            return False
        try:
            return not server_registry.claims()
        except Exception:
            # Unreadable means unknown, and unknown must not read as "nobody asked":
            # erring that way stops a detached run because a file could not be read.
            logger.warning("pool: the detach claims could not be read", exc_info=True)
            return False

    def _consider_stopping(self) -> None:
        """Stop once nothing has needed this process for the whole TTL.

        Measured from when it *first* looked idle, and reset by any activity, so the
        answer is "idle throughout" rather than "idle at some point" — which is the
        difference between stopping a forgotten server and stopping one between two
        turns of a conversation the user is coming back to.

        Ahead of all of that: a server nobody claimed and nobody is watching has no
        reason to live at all, whatever it is in the middle of. See
        :meth:`_nobody_asked_to_keep_it`.
        """
        if self.stop_requested is None:
            return
        if self._nobody_asked_to_keep_it():
            logger.info("pool: no window and no conversation asking to be left "
                        "running — stopping")
            self.request_stop()
            return
        report = self.idle_report()
        if not report["idle"]:
            if self._idle_since is not None:
                logger.info("pool: no longer idle (%s)", "; ".join(report["reasons"]))
            self._idle_since = None
            return
        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now
            logger.info("pool: nothing needs this process; stopping in %.0fs unless "
                        "something does", self.server_idle_ttl)
            return
        if now - self._idle_since >= self.server_idle_ttl:
            logger.info("pool: idle for %.0fs — stopping", now - self._idle_since)
            self.request_stop()

    # ── Addressing a session rather than "the worker" ──────────────────────────

    def is_busy(self, session_id: str | None) -> bool:
        worker = self._workers.get(session_id or "")
        if worker is None:
            return False
        try:
            return worker.is_busy()
        except Exception:
            return False

    def is_parked(self, session_id: str | None) -> bool:
        """Whether this conversation's turn is waiting on the user right now."""
        worker = self._workers.get(session_id or "")
        return worker is not None and getattr(worker, "_pending_prompt", None) is not None

    def set_active(self, session_id: str | None) -> None:
        """Point the pool at the conversation on screen, and tell every worker."""
        self.active_session_id = session_id
        for _sid, worker in self.items():
            worker.active_session_id = session_id

    def set_model(self, model: str) -> list[str]:
        """Switch the served model for every conversation, and for the next one built.

        The model is pool-wide: the backend serves one at a time, so a per-conversation
        model is not a thing that could work. Recorded so a worker built later comes up on
        it, and applied to the detached stand-in too, since that is what the greeting of a
        conversation with no agent reads.
        """
        self.model = model
        self._detached.model = model
        results = self.apply_setting("set_model", model)
        return [r for r in results if r]

    # What the greeting reports, keyed by the setter that would change it. The getter
    # gives the agent's own answer; the recording gives what the user chose before any
    # agent existed. Listed rather than derived because only the pairs that actually
    # correspond belong here — `thinking` is a profile of the model, not a setting.
    _UI_STATE = (
        ("context_mode", "get_context_mode", "set_context_mode"),
        ("enforcement", "get_enforcement", "set_enforcement"),
        ("approval_mode", "get_approval_mode", "set_approval_mode"),
    )

    def ui_state(self, session_id: str | None) -> dict:
        """The settings the front-end draws on connect, for the conversation on screen.

        From its live worker when it has one. When it has none — the ordinary case on a
        fresh connect, since an agent is built on first query — from the stand-in's
        pre-agent defaults, overridden by whatever the user has already chosen in this
        server's lifetime. The override matters because those choices are real: the pool
        replays them onto the agent as soon as one is built, so reporting the bare default
        would show the user a setting they had already changed.
        """
        worker = self.get(session_id)
        source = worker or self._detached
        state: dict[str, Any] = {
            "model": self.model,
            "thinking": source.get_thinking_profile(),
            "temperature": source.get_temperature_state(),
            "agent_ready": source.agent_ready(),
        }
        for key, getter, setter in self._UI_STATE:
            value = getattr(source, getter)()
            if worker is None:
                recorded = self.settings.recorded(setter)
                if recorded:
                    value = recorded[0]
            state[key] = value
        return state

    def apply_setting(self, name: str, *args: Any) -> list[Any]:
        """Record a UI knob and push it to every live worker.

        Both halves matter: the record is what a worker built later starts from, and the
        push is what makes the change take effect in the conversations already running.
        Returns each live worker's own return value, so a caller that reports a rejection
        still can.
        """
        self.settings.record(name, *args)
        results: list[Any] = []
        for _sid, worker in self.items():
            method = getattr(worker, name, None)
            if method is None:
                continue
            try:
                results.append(method(*args))
            except Exception:
                logger.warning("pool: could not apply %s", name, exc_info=True)
        return results

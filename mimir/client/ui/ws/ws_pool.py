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
from datetime import datetime
from typing import Any, Callable, Iterator

from . import server_registry
from ._ws_runtime import _todo_file_for_session
from .event_bus import _EventBus
from .job_scan import has_baseline, scan_all_sessions
from .wake_router import WakeRouter
from .ws_worker import _AgentWorker, _detach_grace
from ...context.todo import load_todo_deps, load_todo_items
from ...core.settings import parse_setting
from ...session.conversation import Conversation, Conversations

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


def idle_stop_reason(ttl_secs: float, *, past: bool) -> str:
    """What this TTL means, in whichever tense the sayer needs.

    Two layers describe it to a user — the notice written into a conversation after the
    process stops, and the panel's own line when they detach onto nothing — and they
    were each spelling it out, with the wording and the formatting drifting apart. One
    author with two tenses, so they cannot disagree about what the clock is.

    What neither may say is a deadline. The clock is measured from when the process
    *first* had nothing to do and is reset by any work, so "it stops in two hours" names
    a moment that does not exist; what ends the process is having had nothing to do
    throughout. The remedy is deliberately absent: this layer knows what it was not
    watching, not what kind of work the user has — a scheduler is something only the
    tools that talk to one know about.
    """
    clock = f"{ttl_secs / 3600:.0f}h"
    if past:
        return (f"MIMIR stopped: it had nothing to do for {clock} together — no turn "
                "in flight, and no run it was watching; any work would have restarted "
                "that clock.")
    return (f"It stops only once it has had nothing to do for about {clock} together "
            "— any work restarts that clock.")


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
    """The choices a newly built worker must start from.

    A worker is built long after the user makes these, so it has to arrive already
    configured: without the record, a conversation started after the user picked "plan
    mode" would quietly come up in agent mode. Every choice is therefore recorded here
    *and* applied to every live worker, and a new worker replays the record once its
    agent exists. The values are validated where they are applied (core.settings), not
    here, so the two cannot drift apart about what a mode may be.
    """

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}              # core.settings name -> value
        # Keyed by (kind, name): one record per server/skill/nudge switched, not one
        # per kind — keyed by method, only the last toggle before the first agent
        # survived to be replayed.
        self.toggles: dict[tuple[str, str], bool] = {}
        self.non_interactive = False

    def replay(self, worker: _AgentWorker) -> None:
        for name, value in self.values.items():
            error = worker.apply_setting(name, value)
            if error:
                logger.warning("pool: could not apply %s to a new worker: %s", name, error)
        for (kind, name), enabled in self.toggles.items():
            try:
                getattr(worker, f"set_{kind}_enabled")(name, enabled)
            except Exception:
                logger.warning("pool: could not switch %s %s on a new worker", kind, name,
                               exc_info=True)
        if self.non_interactive:
            worker.set_non_interactive(True)


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
        # Every conversation of this process, one object each: the connections and
        # this pool write the same one, so nothing has to decide which of them may.
        self.conversations = Conversations()
        self.store = self.conversations.store
        self.wakes = WakeRouter(self)
        self.bus = _EventBus(self, commit=self.commit_turn, durable=self.wakes.consume,
                             steered=self.wakes.steered)

        # Resolved when this process decides it is no longer needed. Awaited beside the
        # signal handlers, so an idle stop and a SIGTERM take the same path out — the
        # one that closes the MCP servers.
        self.stop_requested: asyncio.Event | None = None
        self.server_idle_ttl = _server_idle_ttl()
        # When the whole process first looked idle, or None while it does not. The TTL
        # is measured from here, so a moment of activity restarts it rather than
        # shortening it.
        self._idle_since: float | None = None

    # ── Is this process still needed ──────────────────────────────────────────

    def idle_report(self, *, excluding: Any = None) -> dict:
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

        *excluding* is a subscription to leave out of the count, which is what asks
        that second question properly. A window asking what it would leave behind is
        itself an attachment, so counting it answers "yes, you" every time — which is
        how a detachment came to promise that a run keeps working when there was no run
        and the process was about to stop on idle. Only the asker, not every client: a
        second window of the same workspace *is* keeping this process alive, and
        telling the first that nothing is watching would be the same mistake inverted.
        """
        reasons: list[str] = []
        attached = self.bus.attached(excluding=excluding)
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

    def request_stop(self, headline: str = "") -> None:
        """Ask the serve loop to shut down. Idempotent.

        *headline* is why, when the answer is this process's own: it is recorded into
        every conversation being kept before the loop is asked to go. Recorded here
        rather than at each deciding branch so that "the process decided to stop" is
        one fact in one place — a stop added later cannot forget to say so. The
        user-requested stop passes nothing, and should: the user is there, and already
        has the reasons on screen.
        """
        event = self.stop_requested
        if event is not None and not event.is_set():
            if headline:
                self._record_own_stop(headline)
            event.set()

    # ── Writing a conversation ────────────────────────────────────────────────

    def commit_turn(self, ev: dict, extras: dict) -> dict:
        """Write a finished turn into its conversation, and carry on from there.

        Called by the pump for every answer, before any socket sees it, whether or not
        one is attached: there is one conversation object, so there is one writer, and
        a connection showing the conversation renders what is already written. Writing
        it down is half of a turn landing; the other half is carrying on
        (``WakeRouter.after_turn``), and a process that did only the first half would
        answer the step that finished and stop there.

        Returns what a connection needs to say about it: the conversation's title, the
        interaction it is now waiting on, and the steering that was given a turn.
        """
        session_id = ev.get("session_id")
        conversation = self.conversations.get(session_id)
        if conversation is None:
            logger.warning("commit: the answer of session %s has no conversation to go "
                           "to", session_id)
            return {}
        conversation.commit_answer(ev, extras)
        self.save_conversation(conversation)
        try:
            steering = self.wakes.after_turn(conversation, extras)
        except Exception:
            logger.warning("session %s could not be carried on after its turn",
                           session_id, exc_info=True)
            steering = []
        return {"title": conversation.title or session_id,
                "deferred": extras.get("_deferred"),
                "steer_turn": steering}

    def submit_turn(self, conversation: Conversation, worker: _AgentWorker,
                    text: str) -> bool:
        """Open a turn of *conversation* with *text* and queue it on its own agent.

        The message is written down before the turn is queued, so a reconnect while it
        runs reloads it; when it cannot be, nothing is queued and the message is taken
        back out. Its answer comes back through :meth:`commit_turn`.
        """
        conversation.add_user_message(text)
        if not self.save_conversation(conversation):
            conversation.history.pop()
            conversation.history_full.pop()
            logger.warning("no turn started for session %s: it would not save",
                           conversation.id)
            return False
        worker.submit_query(text, list(conversation.history), session_id=conversation.id)
        return True

    def save_conversation(self, conversation: Conversation) -> bool:
        """Write *conversation* down, with what its agent and its checklist hold now.

        The agent's carried context and mode come from its live worker; with none, the
        conversation keeps what it was saved with. The checklist is read off disk,
        where the agent keeps it.
        """
        record = conversation.record
        worker = self.get(conversation.id)
        if worker is not None:
            try:
                record.carry_context = worker.export_agent_state().get("carry_context", {})
                record.context_mode = worker.setting("context_mode", record.context_mode)
            except Exception:
                logger.warning("session %s: its agent's state could not be read",
                               conversation.id, exc_info=True)
        todo_file = _todo_file_for_session(conversation.id)
        try:
            record.todos = load_todo_items(todo_file)
            record.todo_deps = load_todo_deps(todo_file)
        except Exception:
            logger.warning("session %s: its checklist could not be read",
                           conversation.id, exc_info=True)
        return self.conversations.save(conversation)

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

        evicted = None
        async with self._lock:
            existing = self._workers.get(session_id)
            if existing is not None:
                self._last_use[session_id] = time.monotonic()
                return existing
            pending = self._building.get(session_id)
            # Builds in flight hold a slot too: counted on workers alone, two sessions
            # starting together both saw room and the pool went past its cap.
            if pending is None and len(self._workers) + len(self._building) >= self.cap:
                evicted = self._take_one_idle_locked()
                if evicted is None:
                    return None
            if pending is None:
                pending = asyncio.get_running_loop().create_future()
                self._building[session_id] = pending
                starting = True
            else:
                starting = False
        if evicted is not None:
            await self._close_workers([evicted])

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
        # Every event the worker emits wakes the pump, from whichever thread emits it.
        # Anything it said while being built is picked up by the pump's own tick.
        worker.out_q.on_put = lambda: loop.call_soon_threadsafe(self.bus.nudge)
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
            entry = self._queue[0]
            session_id, submit = entry
            worker = await self.worker_for(session_id)
            if worker is None:
                break
            # Taken out by identity, not by position: the build above yields, and a
            # ``drop_queued`` meanwhile replaces the line — popleft() then removed
            # somebody else's turn and ran one whose session had been deleted.
            try:
                self._queue.remove(entry)
            except ValueError:
                continue
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
        * **holding undrained output** — its last word has not been acted on. Each
          event wakes the pump, which does the whole handover inside one synchronous
          tick; before it, closing the worker discards what it had just said — the
          completion event included — and the agent that event is addressed to with
          it.
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
        conversation = self.conversations._open.get(session_id)
        if conversation is not None and conversation.pending_wakes:
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

    def _take_one_idle_locked(self) -> _AgentWorker | None:
        """Take out the longest-idle releasable worker, to free its slot. Caller holds
        the lock, and closes what it is given once it has let go of it.

        None when nothing may be released — every worker is working, waiting on the
        user, or being read. The caller then queues instead of evicting: taking a slot
        from a conversation that is mid-task to give it to a new one trades a visible
        wait for silent lost work.
        """
        candidates = sorted(
            (sid for sid in self._workers if self.releasable(sid)),
            key=self.idle_for, reverse=True,
        )
        if not candidates:
            return None
        return self._take_locked(candidates[0], reason="a slot was needed")

    async def release_idle(self) -> list[str]:
        """Close every releasable worker idle past the TTL. Returns the sessions closed."""
        closed: list[str] = []
        taken: list[_AgentWorker] = []
        async with self._lock:
            for session_id in list(self._workers):
                if not self.releasable(session_id):
                    continue
                if self.idle_for(session_id) < self.idle_ttl:
                    continue
                taken.append(self._take_locked(session_id, reason="idle"))
                closed.append(session_id)
        await self._close_workers(taken)
        return closed

    async def close(self, session_id: str) -> None:
        """Close one worker outright — its conversation is gone.

        With everything the pool still owed it: a queued turn, wakes, a held check-in.
        Left behind, they kept the bookkeeping of a conversation that no longer exists,
        and a later flush tried to start a turn in it.
        """
        async with self._lock:
            worker = self._take_locked(session_id, reason="its session was deleted")
        self.drop_queued(session_id)
        self.conversations.forget(session_id)
        if worker is not None:
            await self._close_workers([worker])

    def _take_locked(self, session_id: str, *, reason: str) -> _AgentWorker | None:
        """Remove *session_id*'s worker from the pool, freeing its slot. Caller holds the
        lock; the worker is closed after it is released (:meth:`_close_workers`)."""
        worker = self._workers.pop(session_id, None)
        self._last_use.pop(session_id, None)
        if worker is not None:
            logger.info("pool: releasing the agent of session %s (%s)", session_id, reason)
        return worker

    async def _close_workers(self, workers: list[_AgentWorker]) -> None:
        """Close workers taken out of the pool, all at once and outside its lock.

        Outside the lock: a close waits for the MCP servers to go — seconds of
        subprocess teardown — and held across it, the lock kept every other
        conversation from even looking up its own worker. Off the loop for the same
        reason, and together rather than in series: they share nothing.
        """
        if not workers:
            return
        loop = asyncio.get_running_loop()
        results = await asyncio.gather(
            *(loop.run_in_executor(None, worker.aclose) for worker in workers),
            return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("pool: closing an agent failed: %r", result)

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
            workers = [self._take_locked(sid, reason="shutting down")
                       for sid in list(self._workers)]
        await self._close_workers(workers)

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
            self.request_stop(
                "MIMIR stopped: its last window closed and no conversation had "
                "asked for the run to be left going.")
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
            self.request_stop(idle_stop_reason(self.server_idle_ttl, past=True))

    def _record_own_stop(self, headline: str) -> None:
        """Write into every kept conversation that this process stopped itself, and why.

        The counterpart of a run's wake, for the one thing that happens to a
        conversation which the conversation can never see happen. Nothing reached the
        transcript when a server stopped on its own: the decision was printed in the
        process's own log, which the user has no reason to be reading, so a morning
        opened on a thread whose last line was hours old with nothing between — a night
        that reads as lost rather than as a decision taken and recorded.

        Stamped with the wall-clock time, which is the half of the answer the file
        itself cannot give: the gap in the thread is what raises the question, and
        "stopped at 00:38" is what closes it.

        Best effort throughout, and on purpose: this runs on the way out, and a
        conversation whose file will not load is not worth holding a shutdown for.
        """
        stamped = f"⏹ {datetime.now().astimezone():%Y-%m-%d %H:%M} — {headline}"
        try:
            kept = sorted(server_registry.claims())
        except Exception:
            logger.warning("pool: the detach claims could not be read for the stop "
                           "notice", exc_info=True)
            kept = []
        # The conversations that were being kept, or — in the unclaimed case, where by
        # definition there are none — the ones with an agent still here. Those are the
        # threads a user comes back to.
        for session_id in kept or sorted(sid for sid, _w in self.items()):
            conversation = self.conversations.get(session_id)
            if conversation is None:
                continue
            conversation.note(stamped)
            if self.save_conversation(conversation):
                self.bus.record_client_event(
                    session_id, {"type": "server_stopped", "text": headline})

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
        return self.apply_setting("model", model)

    # The settings the greeting reports. `thinking` is not among them: what the panel
    # draws for it is a profile of the model, not a setting.
    _UI_STATE = ("context_mode", "enforcement", "approval_mode")

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
            # No worker means nothing is starting: the agent is built on the first query.
            "agent_ready": worker is None or worker.agent_ready(),
        }
        for key in self._UI_STATE:
            state[key] = self.setting(key, session_id)
        return state

    def setting(self, name: str, session_id: str | None) -> Any:
        """The value of *name* in force for *session_id*'s conversation.

        Its agent's own when it has one; with none, what the user already chose in this
        server's lifetime — replayed onto the agent as soon as one is built, so the bare
        default would show the user a setting they had already changed — else the
        default.
        """
        source = self.get(session_id) or self._detached
        return source.setting(name, self.settings.values.get(name))

    def apply_setting(self, name: str, value: Any) -> list[str]:
        """Record a setting and apply it to every live worker; returns their refusals.

        Both halves matter: the record is what a worker built later starts from, and
        applying it is what makes the change take effect in the conversations already
        running. A value is checked before either: recorded unchecked, a refused value
        was only refused at replay, on an agent being built, where nobody saw it.
        """
        try:
            value = parse_setting(name, value)
        except ValueError as exc:
            return [str(exc)]
        self.settings.values[name] = value
        errors = []
        for _sid, worker in self.items():
            error = worker.apply_setting(name, value)
            if error:
                errors.append(error)
        return errors

    def apply_toggle(self, kind: str, name: str, enabled: bool) -> None:
        """Switch a server, skill or nudge on or off, here and for agents built later."""
        self.settings.toggles[(kind, name)] = enabled
        for _sid, worker in self.items():
            try:
                getattr(worker, f"set_{kind}_enabled")(name, enabled)
            except Exception:
                logger.warning("pool: could not switch %s %s", kind, name, exc_info=True)

    def set_non_interactive(self) -> None:
        """Nobody is at a terminal behind any agent of this pool, from now on."""
        self.settings.non_interactive = True
        for _sid, worker in self.items():
            worker.set_non_interactive(True)

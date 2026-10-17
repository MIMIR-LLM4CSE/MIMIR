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

from .bus import EventBus
from .wakes import WakeRouter
from ..worker.worker import AgentWorker
from ....core.settings import parse_setting
from ....session.conversation import Conversations
from .lifecycle import PoolLifecycle, default_idle_ttl, server_idle_ttl
from .settings_record import SettingsRecord
from .writer import PoolWriter

logger = logging.getLogger(__name__)

# How many agents may be alive at once. The ceiling is the process count: each agent
# spawns its own set of MCP servers, so this multiplies by ~19. Three is a working
# default for a workstation; lower it on a shared login node.
_DEFAULT_CAP = 3


def _cap() -> int:
    try:
        value = int(os.environ.get("MIMIR_MAX_LIVE_SESSIONS", "") or _DEFAULT_CAP)
    except ValueError:
        return _DEFAULT_CAP
    return max(1, value)


class AgentPool(PoolWriter, PoolLifecycle):
    """The live workers, keyed by the session each one works for."""

    def __init__(self, model: str, *, cap: int | None = None,
                 idle_ttl: float | None = None) -> None:
        self.model = model
        self.cap = cap if cap is not None else _cap()
        self.idle_ttl = idle_ttl if idle_ttl is not None else default_idle_ttl()
        # The session the front-end is showing — a property of the view, not of any one
        # worker, which is why it lives on the pool.
        self.active_session_id: str | None = None
        self.settings = SettingsRecord()

        # The stand-in for a conversation with no agent yet, which is the ordinary state
        # until its first query. One per pool, shared: it holds no session's state because
        # it holds no state at all beyond the model.
        self._detached = AgentWorker.detached(model)

        self._workers: dict[str, AgentWorker] = {}
        self._last_use: dict[str, float] = {}
        # Sessions whose worker is being built right now, so two queries arriving
        # together do not build two agents for one conversation.
        self._building: dict[str, asyncio.Future] = {}
        # Submissions waiting for a slot: (session_id, callable taking the worker).
        self._queue: deque[tuple[str, Callable[[AgentWorker], None]]] = deque()
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
        self.bus = EventBus(self, commit=self.commit_turn, durable=self.wakes.consume,
                             steered=self.wakes.steered)

        # Resolved when this process decides it is no longer needed. Awaited beside the
        # signal handlers, so an idle stop and a SIGTERM take the same path out — the
        # one that closes the MCP servers.
        self.stop_requested: asyncio.Event | None = None
        self.server_idle_ttl = server_idle_ttl()
        # When the whole process first looked idle, or None while it does not. The TTL
        # is measured from here, so a moment of activity restarts it rather than
        # shortening it.
        self._idle_since: float | None = None


    # ── Looking up ────────────────────────────────────────────────────────────

    def get(self, session_id: str | None) -> AgentWorker | None:
        """The live worker for *session_id*, or None. Never builds one."""
        if not session_id:
            return None
        worker = self._workers.get(session_id)
        if worker is not None:
            self._last_use[session_id] = time.monotonic()
        return worker

    def worker_or_detached(self, session_id: str | None) -> AgentWorker:
        """The live worker for *session_id*, or the detached stand-in.

        What the session reads for everything that is not a query: the settings behind
        the greeting, a toggle panel, a mode switch typed before anything was asked. Each
        of those has a right answer with no agent — the worker's own getters give it — so a
        stand-in lets every call site that addresses "the conversation on screen" read it
        directly, without a None check of its own.
        """
        return self.get(session_id) or self._detached

    def items(self) -> Iterator[tuple[str, AgentWorker]]:
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
                         on_wait: Callable[[], Any] | None = None) -> AgentWorker | None:
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
                logger.debug("worker_for: a best-effort step failed and was skipped", exc_info=True)
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

    async def _build(self, session_id: str) -> AgentWorker:
        """Construct a worker off the event loop.

        ``AgentWorker.__init__`` blocks until its agent is up — the LLM backend first,
        then ~19 MCP servers — so building it inline would stop the drain loops of every
        other conversation for the duration.
        """
        loop = asyncio.get_running_loop()
        worker = await loop.run_in_executor(
            None, lambda: AgentWorker(self.model, session_id=session_id))
        # Every event the worker emits wakes the pump, from whichever thread emits it.
        # Anything it said while being built is picked up by the pump's own tick.
        worker.out_q.on_put = lambda: loop.call_soon_threadsafe(self.bus.nudge)
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

    def enqueue(self, session_id: str, submit: Callable[[AgentWorker], None]) -> int:
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

    def _take_locked(self, session_id: str, *, reason: str) -> AgentWorker | None:
        """Remove *session_id*'s worker from the pool, freeing its slot. Caller holds the
        lock; the worker is closed after it is released (:meth:`_close_workers`)."""
        worker = self._workers.pop(session_id, None)
        self._last_use.pop(session_id, None)
        if worker is not None:
            logger.info("pool: releasing the agent of session %s (%s)", session_id, reason)
        return worker

    async def _close_workers(self, workers: list[AgentWorker]) -> None:
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


    # ── Addressing a session rather than "the worker" ──────────────────────────

    def is_busy(self, session_id: str | None) -> bool:
        worker = self._workers.get(session_id or "")
        if worker is None:
            return False
        try:
            return worker.is_busy()
        except Exception:
            logger.debug("is_busy: a best-effort step failed and was skipped", exc_info=True)
            return False

    def is_parked(self, session_id: str | None) -> bool:
        """Whether this conversation's turn is waiting on the user right now."""
        worker = self._workers.get(session_id or "")
        return worker is not None and worker.human.awaiting_user()

    def set_active(self, session_id: str | None) -> None:
        """Point the pool at the conversation on screen — what it must never release.

        The one holder of that pointer besides the connection showing it: a worker
        works for its own conversation and has no business knowing which one is read.
        """
        self.active_session_id = session_id

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
                worker.set_enabled(kind, name, enabled)
            except Exception:
                logger.warning("pool: could not switch %s %s", kind, name, exc_info=True)

    def set_non_interactive(self) -> None:
        """Nobody is at a terminal behind any agent of this pool, from now on."""
        self.settings.non_interactive = True
        for _sid, worker in self.items():
            worker.set_non_interactive(True)

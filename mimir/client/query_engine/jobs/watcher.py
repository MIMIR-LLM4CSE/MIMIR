"""Holding the detached runs an agent launched: a conversation's or a sub-agent's.

:class:`RunWatcher` registers a run, polls it to its end through the agent's
``run_tool`` and builds its ``job_complete``; where that event goes is the subclass's
(:meth:`RunWatcher.deliver`). A run is held by a task of its own (:meth:`RunWatcher.register`)
or by the waiting turn (:meth:`RunWatcher.follow`). Its table may be read from another
thread, hence the lock.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import Any, Awaitable, Callable

from ....shared.jobs import job_key_of
from . import poll
from .wakes import job_complete

logger = logging.getLogger(__name__)

RunTool = Callable[..., Awaitable[Any]]

# When check-ins fire, in seconds from the previous one (T+30s, 2.5min, 12.5min,
# 42.5min), then hourly for as long as the run lasts: early mistakes are cheap to catch,
# and the longest runs are never left silent.
_CHECKIN_SCHEDULE = (30.0, 120.0, 600.0, 1800.0)
_CHECKIN_INTERVAL = 3600.0

# Status fields that move on every tick without saying anything about the run. The tail
# is read for its alerts, which are compared instead.
_CHECKIN_VOLATILE = frozenset({"at", "elapsed_s", "log_tail"})


def _checkin_fingerprint(jobs: list[dict]) -> tuple:
    """What a bulletin says, minus the clock: equal fingerprints mean nothing new."""
    return tuple(sorted(
        (str(j.get("job_key")),
         repr(sorted((k, repr(v)) for k, v in j.items() if k not in _CHECKIN_VOLATILE)))
        for j in jobs))


@dataclass
class Watch:
    """A live watch: its task, the run's descriptor, and the last status its poll read.

    The descriptor is what the dispatch guards compare calls against; a check-in reads
    the status instead of polling again.
    """
    task: asyncio.Task
    descriptor: dict
    status: dict = field(default_factory=dict)
    # Alert categories ever seen in the run's log tail, kept once the line scrolls away.
    alerts: dict = field(default_factory=dict)
    # Last (phase, percent) announced, so a poll that learned nothing says nothing.
    progress: tuple = ("", None)
    # Set to read the run's status now rather than at the end of the current wait.
    wake: asyncio.Event = field(default_factory=asyncio.Event)


class RunWatcher:
    """An agent's detached runs, each polled to its end and announced once.

    Lent to the agent as its host's ``watcher`` (``core.ports.AgentHost``).
    """

    first_interval = 5.0
    max_interval = 30.0
    # None reads poll's own default at each poll.
    call_timeout: float | None = None
    unreadable_limit: int | None = None

    def __init__(self) -> None:
        self.watches: dict[str, Watch] = {}
        self._lock = threading.Lock()
        # The check-in cycle covering every run of this agent, or None between waves. One
        # per agent rather than one per run: runs launched in the same step share a
        # schedule and report together.
        self._checkin_task: asyncio.Task | None = None
        # Fingerprint of the last bulletin sent, so one with nothing new is not.
        self._checkin_last: tuple | None = None

    # ── what each watcher supplies ───────────────────────────────────────────────

    def tool_runner(self) -> RunTool | None:
        """The agent's ``run_tool``, asked for when a watch first polls."""
        raise NotImplementedError

    def deliver(self, event: dict) -> None:
        """Hand a ``job_complete`` or a ``job_checkin`` to whoever answers for the runs."""
        raise NotImplementedError

    def cancel_flag(self) -> Any:
        """An event that, once set, ends every poll (the user's Stop), or None."""
        return None

    def accepts(self) -> bool:
        """Whether a new run may be taken on now."""
        return True

    def session_for(self, owner: str | None) -> str | None:
        """The session a run registered for *owner* wakes."""
        return owner

    def adopt(self, descriptor: dict, session_id: str | None) -> None:
        """Complete a descriptor as it is taken on (its origin, say)."""

    def on_tick(self, job_key: str, watch: Watch, tick: poll.Tick,
                session_id: str | None) -> None:
        """Called after each poll, once ``watch.status`` holds what it read."""

    # ── the agent's hooks ────────────────────────────────────────────────────────

    def register(self, descriptor: dict, owner: str | None = None) -> bool:
        """Watch a detached run; True when one is holding it, False means wait it out in-turn.

        Deduplicated by ``job_key``; the wake goes to :meth:`session_for` *owner*.
        """
        if not isinstance(descriptor, dict):
            logger.warning("background-job registration refused: descriptor is %s, "
                           "not a dict", type(descriptor).__name__)
            return False
        job_key = job_key_of(descriptor)
        if not job_key:
            logger.warning("background-job registration refused: descriptor carries "
                           "neither 'job_key' nor 'run_dir' (keys: %s)", sorted(descriptor))
            return False
        if not (descriptor.get("status_op") or {}).get("tool"):
            # Nothing to poll is nothing to promise; False sends the caller back to
            # waiting in-turn.
            logger.warning("background-job registration refused for %r: its descriptor "
                           "names no status op", job_key)
            return False
        session_id = self.session_for(owner)
        with self._lock:
            if not self.accepts():
                return False
            existing = self.watches.get(job_key)
            if existing is not None and not existing.task.done():
                return True   # already watched
            self.adopt(descriptor, session_id)
            try:
                # get_running_loop: a task on a loop that is not running never polls,
                # while registration would still report success.
                task = asyncio.get_running_loop().create_task(
                    self._watch_job(job_key, descriptor, session_id))
            except RuntimeError:
                logger.warning("background-job registration failed for %r: no running "
                               "event loop to host the watcher", job_key, exc_info=True)
                return False
            self.watches[job_key] = Watch(task, descriptor)
        task.add_done_callback(self.log_death)
        logger.info("background job %r of session %s is now watched by %r",
                    job_key, session_id, descriptor["status_op"]["tool"])
        self._start_checkins(session_id)
        return True

    async def follow(self, descriptor: dict) -> None:
        """Watch a run to its end in the caller's own task: a turn that waits it out.

        No check-ins: the turn is already waiting on it.
        """
        job_key = job_key_of(descriptor) or "?"
        with self._lock:
            self.watches[job_key] = Watch(asyncio.current_task(), descriptor)  # type: ignore[arg-type]
        await self._watch_job(job_key, descriptor, self.session_for(None))

    def watched(self) -> list[dict]:
        """Descriptors of the runs a live watch holds (what the dispatch guards consult)."""
        with self._lock:
            return [w.descriptor for w in self.watches.values() if not w.task.done()]

    def cancel_op(self, job_key: str) -> dict | None:
        """The call that stops *job_key*, as its descriptor names it; None when it names none."""
        with self._lock:
            watch = self.watches.get(job_key)
        if watch is None or watch.task.done():
            return None
        found = watch.descriptor.get("cancel_op")
        return found if isinstance(found, dict) and found.get("tool") else None

    def poll_now(self, job_key: str) -> None:
        """Have *job_key*'s watch read its status at once. On the loop the watch runs on."""
        with self._lock:
            watch = self.watches.get(job_key)
        if watch is not None:
            watch.wake.set()

    def watched_job_keys(self) -> list[str]:
        """Keys of the runs still in flight."""
        with self._lock:
            return [k for k, w in self.watches.items() if not w.task.done()]

    def tasks(self) -> list[asyncio.Task]:
        """Every task this holds — the watches and the check-in cycle — for shutdown."""
        with self._lock:
            tasks = [w.task for w in self.watches.values()]
        if self._checkin_task is not None:
            tasks.append(self._checkin_task)
        return tasks

    async def close(self) -> None:
        """Cancel every watch and wait for it, on the loop that runs them."""
        tasks = self.tasks()
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("a cancelled watch raised on the way out", exc_info=True)

    @staticmethod
    def log_death(task: asyncio.Task) -> None:
        """Log a task of this watcher that ended on an exception (nothing awaits them)."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("background-job task died: %r — any run it was holding will "
                           "not wake anyone", exc, exc_info=exc)

    # ── the check-in cycle ───────────────────────────────────────────────────────

    def _start_checkins(self, session_id: str | None) -> None:
        """Start a check-in cycle for a new wave of runs, replacing a stale one.

        A job joining a running wave inherits its schedule.
        """
        wave_starts = len(self.watched_job_keys()) <= 1
        if self._checkin_task is not None and not self._checkin_task.done():
            if not wave_starts:
                return
            self._checkin_task.cancel()
        self._checkin_last = None
        try:
            self._checkin_task = asyncio.get_running_loop().create_task(
                self._checkin_cycle(session_id))
        except RuntimeError:
            # Same precondition as the watcher's, and the same honesty: without a loop
            # there are no check-ins, and the run is still watched. Not a failure of the
            # registration — the completion wake does not depend on this.
            logger.warning("background check-ins unavailable for session %s: no running "
                           "event loop to host the cycle", session_id)
            self._checkin_task = None
            return
        self._checkin_task.add_done_callback(self.log_death)

    async def _checkin_cycle(self, session_id: str | None) -> None:
        """Emit ``job_checkin`` down ``_CHECKIN_SCHEDULE``, then hourly, while live runs remain.

        Reads what the watchers already know; a point with nothing new says nothing.
        """
        try:
            for delay in chain(_CHECKIN_SCHEDULE, repeat(_CHECKIN_INTERVAL)):
                await asyncio.sleep(delay)
                if not self._emit_checkin(session_id):
                    return
        finally:
            # Only if the slot still holds *this* cycle. A new wave cancels the old one
            # and puts its own task in the slot, and cancellation is delivered after
            # that — so an unconditional clear here would erase the live cycle's
            # reference and let the next launch start a second one beside it.
            if self._checkin_task is asyncio.current_task():
                self._checkin_task = None

    def _emit_checkin(self, session_id: str | None) -> bool:
        """Send a bulletin if it says something the last one did not. False: no live run."""
        with self._lock:
            jobs = [
                {"job_key": key, "kind": w.descriptor.get("kind"),
                 "server": w.descriptor.get("server"), **w.status}
                for key, w in self.watches.items() if not w.task.done()
            ]
        if not jobs:
            return False
        fingerprint = _checkin_fingerprint(jobs)
        if fingerprint == self._checkin_last:
            return True
        self._checkin_last = fingerprint
        self.deliver({"type": "job_checkin", "session_id": session_id, "jobs": jobs})
        return True

    # ── one watch ────────────────────────────────────────────────────────────────

    async def _watch_job(self, job_key: str, descriptor: dict,
                         session_id: str | None = None) -> None:
        """Poll a run to its end, then deliver its ``job_complete``; leave the table on any exit."""
        try:
            await self._poll_to_completion(job_key, descriptor, session_id)
        finally:
            # A finished, cancelled or failed watch holds nothing: an entry left behind
            # would say a run is in flight when it is over.
            with self._lock:
                self.watches.pop(job_key, None)

    async def _poll_to_completion(self, job_key: str, descriptor: dict,
                                  session_id: str | None) -> None:
        status_op = descriptor.get("status_op") or {}
        summary_op = descriptor.get("summary_op") or {}
        if not status_op.get("tool"):
            return
        run_tool = self.tool_runner()
        if run_tool is None:
            logger.warning("job %r cannot be polled: the agent has no tool runner", job_key)
            return
        call_timeout = (self.call_timeout if self.call_timeout is not None
                        else poll.STATUS_CALL_TIMEOUT)
        with self._lock:
            held = self.watches.get(job_key)

        def _tick(tick: poll.Tick) -> None:
            with self._lock:
                watch = self.watches.get(job_key)
            if watch is None:
                return
            fresh = {k: v for k, v in poll.log_alerts(tick.fields.get("log_tail")).items()
                     if k not in watch.alerts}
            watch.alerts.update(fresh)
            # The state goes with it: "running" and "unreadable" are what a bulletin most
            # needs to tell apart, and only the poll knows which it is looking at.
            watch.status = {**tick.fields, "state": tick.state or "unreadable",
                            "phase": tick.phase, "percent": tick.percent,
                            "at": time.monotonic()}
            if watch.alerts:
                watch.status["alerts"] = dict(watch.alerts)
            # A run gone wrong is not left waiting for the next point of the schedule.
            if fresh:
                self._emit_checkin(session_id)
            self.on_tick(job_key, watch, tick, session_id)

        state, reason = await poll.poll_until_terminal(
            run_tool, status_op,
            interval=self.first_interval, max_interval=self.max_interval,
            call_timeout=call_timeout,
            unreadable_limit=(self.unreadable_limit if self.unreadable_limit is not None
                              else poll.UNREADABLE_POLL_LIMIT),
            on_tick=_tick, label=job_key, cancel_flag=self.cancel_flag(),
            wake=held.wake if held is not None else None)
        summary = await poll.read_summary(run_tool, summary_op, call_timeout=call_timeout,
                                          label=job_key)
        logger.info("background job %r of session %s finished (%s)",
                    job_key, session_id, state)
        self.deliver(job_complete(descriptor, state, session_id=session_id,
                                  summary=summary, reason=reason))

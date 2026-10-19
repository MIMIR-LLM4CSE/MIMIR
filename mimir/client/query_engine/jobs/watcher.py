"""Holding the detached runs an agent launched: a conversation's or a sub-agent's.

:class:`RunWatcher` registers a run, polls it to its end through the agent's
``run_tool`` and builds its ``job_complete``; where that event goes is the subclass's
(:meth:`RunWatcher.deliver`). Its table may be read from another thread, hence the lock.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ....shared.jobs import job_key_of
from . import poll
from .wakes import job_complete

logger = logging.getLogger(__name__)

RunTool = Callable[..., Awaitable[Any]]


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


class RunWatcher:
    """An agent's detached runs, each polled to its end and announced once.

    Installed on the agent as ``register_background_job`` (:meth:`register`) and
    ``watched_background_jobs`` (:meth:`watched`).
    """

    first_interval = 5.0
    max_interval = 30.0
    # None reads poll's own default at each poll.
    call_timeout: float | None = None
    unreadable_limit: int | None = None

    def __init__(self) -> None:
        self.watches: dict[str, Watch] = {}
        self._lock = threading.Lock()

    # ── what each watcher supplies ───────────────────────────────────────────────

    def tool_runner(self) -> RunTool | None:
        """The agent's ``run_tool``, asked for when a watch first polls."""
        raise NotImplementedError

    def deliver(self, job_key: str, event: dict) -> None:
        """Hand a finished run's ``job_complete`` to whoever answers for it."""
        raise NotImplementedError

    def accepts(self) -> bool:
        """Whether a new run may be taken on now."""
        return True

    def session_for(self, owner: str | None) -> str | None:
        """The session a run registered for *owner* wakes."""
        return owner

    def adopt(self, descriptor: dict, session_id: str | None) -> None:
        """Complete a descriptor as it is taken on (its origin, say)."""

    def on_registered(self, job_key: str, descriptor: dict, session_id: str | None) -> None:
        """Called once a new watch is running."""

    def on_tick(self, job_key: str, watch: Watch, tick: poll.Tick, fresh_alerts: dict,
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
        self.on_registered(job_key, descriptor, session_id)
        return True

    def watched(self) -> list[dict]:
        """Descriptors of the runs a live watch holds (what the dispatch guards consult)."""
        with self._lock:
            return [w.descriptor for w in self.watches.values() if not w.task.done()]

    def watched_job_keys(self) -> list[str]:
        """Keys of the runs still in flight."""
        with self._lock:
            return [k for k, w in self.watches.items() if not w.task.done()]

    def tasks(self) -> list[asyncio.Task]:
        """Every watch's task, for shutdown."""
        with self._lock:
            return [w.task for w in self.watches.values()]

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
            self.on_tick(job_key, watch, tick, fresh, session_id)

        state, reason = await poll.poll_until_terminal(
            run_tool, status_op,
            interval=self.first_interval, max_interval=self.max_interval,
            call_timeout=call_timeout,
            unreadable_limit=(self.unreadable_limit if self.unreadable_limit is not None
                              else poll.UNREADABLE_POLL_LIMIT),
            on_tick=_tick, label=job_key)
        summary = await poll.read_summary(run_tool, summary_op, call_timeout=call_timeout,
                                          label=job_key)
        logger.info("background job %r of session %s finished (%s)",
                    job_key, session_id, state)
        self.deliver(job_key, job_complete(descriptor, state, session_id=session_id,
                                           summary=summary, reason=reason))

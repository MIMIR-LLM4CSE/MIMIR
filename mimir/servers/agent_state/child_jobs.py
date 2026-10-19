"""A sub-agent's own watcher for the detached runs it launched.

It installs the child's ``register_background_job``/``watched_background_jobs`` pair. A
finished run reaches the child as a steer (``poll_steer``), in ``wake_text``'s words. The
child's loop is its own: only :meth:`ChildWatcher.handover` may be called from another
thread; :meth:`ChildWatcher.close` cancels, on the child's loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

from mimir.client.query_engine.jobs import poll as job_poll
from ...shared.jobs import job_key_of
from mimir.client.query_engine.jobs.wakes import job_complete, wake_text

logger = logging.getLogger(__name__)

# A child asks more often than the parent's watcher does. Its runs are steps of a task it
# is holding open against a clock, so hearing a minute late costs it a minute of budget,
# where the parent's watcher is holding a run across turns and can afford to be patient.
# Still immeasurably cheaper than a model call per poll, which is the point of a watcher.
_FIRST_INTERVAL = 3.0
_MAX_INTERVAL = 20.0
_CALL_TIMEOUT = 60.0
_UNREADABLE_LIMIT = 5


class ChildWatcher:
    """Holds a sub-agent's detached runs, and answers for them when it ends.

    Constructed per child, on the child's own loop, and installed on its agent as the
    three hooks the client's own watcher installs plus the steer channel its resume
    arrives through.
    """

    def __init__(self, *, agent, origin: dict, on_activity=None) -> None:
        # The agent itself, resolved to its tool runner only when a watch actually
        # polls: that is the moment the capability is needed, and asking for it at
        # construction would make a watcher impossible to install on an agent that has
        # not finished being one.
        self._agent = agent
        self._origin = dict(origin or {})
        self._on_activity = on_activity
        self._tasks: dict[str, asyncio.Task] = {}
        self._descriptors: dict[str, dict] = {}
        # Written by the watch tasks and drained by the agent loop — both on the child's
        # loop, so the deque needs no lock of its own. The descriptors do: handover() may
        # read them from the caller's thread.
        self._steers: deque[str] = deque()
        self._lock = threading.Lock()
        self._handed_over = False

    # ── the hooks the agent reads ────────────────────────────────────────────────

    def register(self, descriptor: dict, owner: str | None = None) -> bool:
        """Watch a detached run (the child's ``register_background_job``); False means wait it out in-turn.

        *owner* is ignored: the child is the only reader.
        """
        if not isinstance(descriptor, dict):
            logger.warning("child watcher: refused a descriptor that is not a mapping")
            return False
        job_key = job_key_of(descriptor)
        if not job_key:
            logger.warning("child watcher: refused a descriptor naming no job")
            return False
        if not ((descriptor.get("status_op") or {}).get("tool")):
            # Nothing to poll, so nothing can be promised. The client's watcher drops
            # such a descriptor a step later; refusing it here says so to the caller,
            # which is what sends it back to waiting in-turn.
            logger.warning("child watcher: refused job %r, its descriptor names no "
                           "status op", job_key)
            return False
        with self._lock:
            if self._handed_over:
                return False
            if job_key in self._tasks and not self._tasks[job_key].done():
                return True   # already watched
            self._descriptors[job_key] = dict(descriptor)
        try:
            task = asyncio.get_running_loop().create_task(self._watch(job_key))
        except RuntimeError:
            # Off the child's loop there is no task to create, and a watcher that
            # reported success without polling anything is the one failure the client's
            # hook documents at length.
            logger.warning("child watcher: no running loop; job %r is not watched",
                           job_key)
            with self._lock:
                self._descriptors.pop(job_key, None)
            return False
        self._tasks[job_key] = task
        task.add_done_callback(lambda t: self._finished(job_key, t))
        return True

    def live_descriptors(self) -> list[dict]:
        """The runs still being watched. The child's ``watched_background_jobs``.

        What the dispatch guards consult to refuse a status poll on a run a watcher is
        already holding.
        """
        with self._lock:
            return [dict(d) for key, d in self._descriptors.items()
                    if key in self._tasks and not self._tasks[key].done()]

    def drain_steers(self) -> list[str]:
        """Pop every wake waiting for the child. The child's ``poll_steer``."""
        out = []
        while self._steers:
            out.append(self._steers.popleft())
        return out

    # ── answering for the runs when the child ends ───────────────────────────────

    def handover(self) -> list[dict]:
        """The runs still going, as descriptors the parent can poll; stops accepting new ones.

        Called from the caller's thread, cancels nothing. Ops are given the child's ``run_dir``,
        or the parent's poll would resolve the key against its own session and read ``unknown``.
        """
        with self._lock:
            if self._handed_over:
                # Idempotent, because two callers reach this on the overrun path: the
                # caller out of patience reads it from its own thread, and the child's
                # own finally reads it a moment later when its thread unwinds. The
                # first owns the handover; a second list would have the parent adopt
                # the same runs twice.
                return []
            self._handed_over = True
            live = [dict(d) for key, d in self._descriptors.items()
                    if key in self._tasks and not self._tasks[key].done()]
        return [self._with_origin(d) for d in live]

    async def close(self) -> None:
        """Drop the watches. Called from the child's own loop, in its ``finally``.

        A task still polling when ``asyncio.run`` returns is the one way a child's thread
        can be left hanging, so this is not optional housekeeping.
        """
        with self._lock:
            self._handed_over = True
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.debug("child watcher: a cancelled watch raised on the way out",
                             exc_info=True)

    # ── internals ───────────────────────────────────────────────────────────────

    def _with_origin(self, descriptor: dict) -> dict:
        """*descriptor* as handed to the parent: it says which child it came from. Its ops
        already name the run's directory, so they resolve from the parent's session."""
        out = dict(descriptor)
        out.setdefault("origin", dict(self._origin))
        return out

    def _finished(self, job_key: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("child watcher: the watch on job %r raised", job_key,
                           exc_info=exc)

    async def _watch(self, job_key: str) -> None:
        run_tool = getattr(self._agent, "run_tool", None)
        if run_tool is None:
            logger.warning("child watcher: job %r cannot be polled, the agent has no "
                           "tool runner", job_key)
            return
        with self._lock:
            descriptor = dict(self._descriptors.get(job_key) or {})
        status_op = descriptor.get("status_op") or {}
        summary_op = descriptor.get("summary_op") or {}
        state, reason = await job_poll.poll_until_terminal(
            run_tool, status_op,
            interval=_FIRST_INTERVAL, max_interval=_MAX_INTERVAL,
            call_timeout=_CALL_TIMEOUT, unreadable_limit=_UNREADABLE_LIMIT,
            label=job_key)
        summary = await job_poll.read_summary(
            run_tool, summary_op, call_timeout=_CALL_TIMEOUT, label=job_key)
        event = job_complete(descriptor, state, summary=summary, reason=reason)
        with self._lock:
            handed_over = self._handed_over
        if handed_over:
            # Its caller has already been told this run is the parent's. Steering the
            # child now would have two agents answering for one run, and the child's
            # answer has already been read.
            return
        self._steers.append(wake_text(event))
        if self._on_activity is not None:
            try:
                self._on_activity({"type": "status",
                                   "text": f"background job {job_key}: {state}"})
            except Exception:
                pass   # a child's activity line must not fail its run

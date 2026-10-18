"""A sub-agent's watcher, for the detached runs it launched itself.

The hook pair the client installs (``register_background_job`` /
``watched_background_jobs``) is set on the agent the WebSocket worker owns; a child's
agent is built here, and gets its own pair from this module. Without it a child would
poll its runs in-turn, inside its wall budget, instead of preparing its next step.

**The resume is a steer, not an event.** The parent's watcher puts ``job_complete`` on a
queue, the event bus claims it and a turn is submitted with the wake folded in. A child has
none of that: no ``out_q``, no bus, and no turn loop to submit to — it is one
``agent.run(...)`` call. What it does have is ``poll_steer``, read at the top of every step
by ``agent_loop._drain_steer`` and documented there as optional precisely for callers like
this one. So a finished run reaches the child's reasoning the same way a user's mid-turn
message reaches the parent's: as a user turn at the next step boundary, in the words
``jobs.wakes.wake_text`` writes for every wake in the system.

**Two threads, two methods.** The child runs ``asyncio.run()`` on a thread of its own, so
its watch tasks live on a loop no other thread may touch — the same constraint
``_abandon_child`` states for the cancel flag. :meth:`ChildWatcher.handover` is therefore
the only method a caller on another thread may use: it reads descriptors under a lock and
cancels nothing. :meth:`ChildWatcher.close` does the cancelling, and is called from the
child's own ``finally``, on its own loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque

from mimir.client.query_engine.jobs import poll as job_poll
from mimir.client.query_engine.jobs.wakes import wake_text

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
        """Put a watcher on a detached run. The child's ``register_background_job``.

        The return value is the whole contract, as it is for the client's own hook:
        ``False`` means nothing is holding the run, and the caller must wait it out
        in-turn instead. Refusing silently is what makes a run finish into nothing.

        *owner* is accepted and ignored. The client's hook takes it to route a wake to a
        conversation; here there is only one reader, and it is this child.
        """
        if not isinstance(descriptor, dict):
            logger.warning("child watcher: refused a descriptor that is not a mapping")
            return False
        job_key = str(descriptor.get("job_key") or descriptor.get("run_dir") or "")
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
        """The runs still going, as descriptors the *parent* can poll.

        Called when the child is out of time, from the caller's thread. Cancels nothing:
        the tasks live on the child's loop and only :meth:`close` may touch them. What it
        does is stop accepting new runs and hand back what is in flight, so the client can
        adopt it for the parent before the thread is let go.

        A descriptor is rewritten by shape, never by tool name: one that carries a
        ``run_dir`` whose ops do not name one gets it added to their arguments. A child's
        job lives under the child's session, and a status op resolving a bare key against
        the *parent's* session finds nothing — which it reports as ``unknown``, a terminal
        state. Handing the parent an unqualified key would have it announce a healthy run
        as lost seconds after adopting it.
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
        return [self._parent_resolvable(d) for d in live]

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

    def _parent_resolvable(self, descriptor: dict) -> dict:
        run_dir = descriptor.get("run_dir")
        out = dict(descriptor)
        out.setdefault("origin", dict(self._origin))
        if not run_dir:
            return out
        for op_name in ("status_op", "summary_op"):
            op = descriptor.get(op_name)
            if not isinstance(op, dict) or not op.get("tool"):
                continue
            args = dict(op.get("args") or {})
            if not args.get("run_dir"):
                args["run_dir"] = run_dir
            out[op_name] = {**op, "args": args}
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
        event = {
            "type": "job_complete",
            "job_key": job_key,
            "server": descriptor.get("server"),
            "kind": descriptor.get("kind"),
            "state": state,
            "summary": summary,
            "status_op": status_op,
            "summary_op": summary_op,
            "reason": reason if state == "unknown" else "",
        }
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

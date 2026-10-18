"""The runs a conversation launched and left running: watched, reported, checked on.

A watcher polls a run's status op until it ends and puts a ``job_complete`` on the
outbox for the pool's ``WakeRouter``; a check-in cycle sends bulletins meanwhile.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import TYPE_CHECKING

from ....session.jobs_on_disk import (
    establish_baseline,
    has_baseline,
    scan_session,
    subagent_sessions,
)
from ....query_engine.jobs import poll

if TYPE_CHECKING:
    from .worker import AgentWorker

logger = logging.getLogger(__name__)


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
class _Watch:
    """A live watcher: its task, the job's descriptor, and the last ``status`` it learned.

    The descriptor is what the dispatch guard compares calls against; check-ins read the
    status instead of polling again.
    """
    task: asyncio.Task
    descriptor: dict
    status: dict = field(default_factory=dict)
    # Alert categories ever seen in this run's log tail, kept once the line scrolls away.
    alerts: dict = field(default_factory=dict)



class JobWatcher:
    """One conversation's background runs and its check-in cycle."""

    def __init__(self, worker: "AgentWorker") -> None:
        self.worker = worker
        # job_key -> _Watch(task, descriptor), polling a detached run to completion.
        # Written by ``register`` (the agent's hook) and read back by ``watched``,
        # which the dispatch guard uses. Only ever touched on the worker loop.
        self.watches: dict[str, _Watch] = {}
        # The check-in cycle covering every job of this conversation, or None between
        # waves. One per conversation rather than one per job: three jobs launched in
        # the same step share a schedule and report together instead of waking it
        # three times a minute apart.
        self._checkin_task: asyncio.Task | None = None
        # Fingerprint of the last bulletin sent, so one with nothing new is not.
        self._checkin_last: tuple | None = None

    def tasks(self) -> list[asyncio.Task]:
        """Every task this holds — watchers and the check-in cycle — for shutdown."""
        tasks = [w.task for w in self.watches.values()]
        if self._checkin_task is not None:
            tasks.append(self._checkin_task)
        return tasks

    def register(self, descriptor: dict, owner: str | None = None) -> bool:
        """Register a watcher for a detached run (the agent's hook); True when one is active.

        Deduplicated by ``job_key``. The wake goes to *owner*, else the session of the running
        query.
        """
        if not isinstance(descriptor, dict):
            logger.warning("background-job registration refused: descriptor is %s, "
                           "not a dict", type(descriptor).__name__)
            return False
        job_key = str(descriptor.get("job_key") or descriptor.get("run_dir") or "")
        if not job_key:
            logger.warning("background-job registration refused: descriptor carries "
                           "neither 'job_key' nor 'run_dir' (keys: %s)",
                           sorted(descriptor))
            return False
        existing = self.watches.get(job_key)
        if existing is not None and not existing.task.done():
            return True  # already watched
        session_id = owner or self.worker.query_session_id or self.worker.session_id
        # Whose run this is, which is not the same question as who gets woken: a run a
        # sub-agent launched and handed over is watched for the parent and still named
        # after the child. setdefault, never an assignment: a descriptor that arrived
        # with an origin arrived with the only end that knew it.
        descriptor.setdefault("origin", {"kind": "main", "label": "",
                                         "session": session_id})
        try:
            # get_running_loop, not get_event_loop: the latter can hand back a loop
            # that is not running, and a task created on one of those never polls
            # anything while registration still reports success. That combination is
            # the worst of both — the model is promised a resume, and nothing is
            # holding the run. Registering is only ever done from the worker loop, so
            # requiring a running one states the real precondition.
            task = asyncio.get_running_loop().create_task(
                self._watch_job(job_key, descriptor, session_id))
        except RuntimeError:
            # No running loop to host the watcher. Logged rather than swallowed: this
            # is the point where a launched run stops being tracked by anything, and
            # every minute after it is a run finishing into silence.
            logger.warning("background-job registration failed for %r: no running "
                           "event loop to host the watcher", job_key, exc_info=True)
            return False
        task.add_done_callback(self._watcher_died)
        self.watches[job_key] = _Watch(task, descriptor)
        logger.info("background job %r of session %s is now watched by %r",
                    job_key, session_id, (descriptor.get("status_op") or {}).get("tool"))
        self._start_checkins(session_id)
        return True

    def rearm_detached_jobs(self) -> dict:
        """Put watchers back on this session's runs and report those that ended; return what it did.

        Only runs no watcher holds; a restart loses the promise to report, never the run.
        """
        session_id = self.worker.session_id
        if not session_id:
            return {"rearmed": [], "reported": []}
        # This session's own runs and its sub-agents'. A child's watcher goes with the
        # child, and a child killed outright hands nothing over, so its runs are the
        # parent's to pick back up — the person is watching the parent, and nothing can
        # load a sub-session as a conversation.
        jobs: list = []
        for sid in (session_id, *subagent_sessions(session_id)):
            found = scan_session(sid)
            if not has_baseline(sid):
                # First time this session has been looked at: everything already
                # finished was reported by whatever was watching it, long before
                # markers existed. Claiming that history as a backlog of wakes would
                # wake a conversation for every build it ever ran. Per session, so a
                # child seen for the first time brings no backlog of its own either.
                establish_baseline(sid, found)
                found = [job for job in found if job.live]
            jobs.extend(found)
        rearmed: list[str] = []
        reported: list[str] = []
        for job in jobs:
            existing = self.watches.get(job.job_key)
            if existing is not None and not existing.task.done():
                continue   # a watcher is already holding it
            if job.live:
                if self.register(job.descriptor(), owner=session_id):
                    rearmed.append(job.job_key)
                continue
            # Ended with nobody watching. Emitted as the watcher would have — the
            # descriptor's own server and kind included — because a consumer must not be
            # able to tell a run reported late from one reported on time, and the wake
            # text names the kind. Settled by that consumer, not here: see
            # ``jobs_on_disk.mark_wakes_reported``.
            descriptor = job.descriptor()
            self.worker.out_q.put({
                "type": "job_complete",
                "job_key": job.job_key,
                "server": descriptor.get("server"),
                "kind": descriptor.get("kind"),
                "state": job.state,
                "session_id": session_id,
                "run_session": job.session_id,
                "summary": {"command": job.command, "exit_code": job.exit_code},
                "status_op": job.status_op(),
            })
            reported.append(job.job_key)
        if rearmed or reported:
            logger.info("job scan: session %s re-armed %d run(s) and reported %d "
                        "that had ended", session_id, len(rearmed), len(reported))
        return {"rearmed": rearmed, "reported": reported}

    @staticmethod
    def _watcher_died(task: asyncio.Task) -> None:
        """Log a watcher or check-in task that ended on an exception (nothing awaits them)."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("background-job task died: %r — any run it was holding will "
                           "not wake anyone", exc, exc_info=exc)

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
        self._checkin_task.add_done_callback(self._watcher_died)

    async def _checkin_cycle(self, session_id: str | None) -> None:
        """Emit ``job_checkin`` down ``_CHECKIN_SCHEDULE``, then hourly, while the conversation has live runs.

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
        self.worker.out_q.put({
            "type":       "job_checkin",
            "session_id": session_id,
            "jobs":       jobs,
        })
        return True

    def watched_job_keys(self) -> list[str]:
        """Keys of the runs this conversation still has in flight."""
        return [k for k, w in self.watches.items() if not w.task.done()]

    def watched(self) -> list[dict]:
        """Descriptors of the runs a live watcher holds (the agent's hook for the dispatch guard)."""
        return [w.descriptor for w in self.watches.values()
                if isinstance(w.descriptor, dict) and not w.task.done()]

    async def _watch_job(self, job_key: str, descriptor: dict,
                         session_id: str | None = None) -> None:
        """Poll a detached run to completion (:func:`poll.poll_until_terminal`), then emit ``job_complete``.

        A task on the worker loop that outlives the launching turn; the event carries the
        descriptor's own ops.
        """
        try:
            await self._poll_to_completion(job_key, descriptor, session_id)
        finally:
            # One exit for every way this ends — a terminal state, a cancellation at
            # shutdown, a status op that was never there, or an exception nothing
            # catches. A run is in this table because something is still asking about
            # it, so an entry that outlives its task is the table saying what is not
            # true. Every reader already filters a finished task out; this is what
            # keeps the table itself honest.
            self.watches.pop(job_key, None)

    async def _poll_to_completion(self, job_key: str, descriptor: dict,
                                  session_id: str | None) -> None:
        """Poll one run to its end and announce it. See :meth:`_watch_job`."""
        status_op  = descriptor.get("status_op") or {}
        summary_op = descriptor.get("summary_op") or {}
        if not status_op.get("tool"):
            return

        # Last (phase, percent) announced, so a poll that learned nothing says nothing.
        last_reported: tuple = ("", None)

        def _tick(tick: poll.Tick) -> None:
            nonlocal last_reported
            # What a check-in reports, written by the poll that is running anyway. The
            # state goes with it: "running" and "its status stopped being readable" are
            # the two things a bulletin most needs to tell apart, and only the poll knows
            # which one it is looking at.
            watch = self.watches.get(job_key)
            if watch is not None:
                fresh = {k: v for k, v in
                         poll.log_alerts(tick.fields.get("log_tail")).items()
                         if k not in watch.alerts}
                watch.alerts.update(fresh)
                watch.status = {**tick.fields,
                                "state": tick.state or "unreadable",
                                "phase": tick.phase, "percent": tick.percent,
                                "at": time.monotonic()}
                if watch.alerts:
                    watch.status["alerts"] = dict(watch.alerts)
                # A run gone wrong is not left waiting for the next point of the schedule.
                if fresh:
                    self._emit_checkin(session_id)
            # Starts at ("", None), so a run that never reports is never sent, while one
            # whose count disappears is — a retraction is news too.
            if (tick.phase, tick.percent) != last_reported:
                last_reported = (tick.phase, tick.percent)
                self.worker.out_q.put({
                    "type":       "job_progress",
                    "job_key":    job_key,
                    "phase":      tick.phase,
                    "percent":    tick.percent,
                    "session_id": session_id,
                })

        state, reason = await poll.poll_until_terminal(
            self.worker.agent.run_tool, status_op,
            interval=5.0, max_interval=30.0,
            call_timeout=poll.STATUS_CALL_TIMEOUT,
            unreadable_limit=poll.UNREADABLE_POLL_LIMIT,
            on_tick=_tick, label=job_key)

        summary = await poll.read_summary(
            self.worker.agent.run_tool, summary_op,
            call_timeout=poll.STATUS_CALL_TIMEOUT, label=job_key)

        logger.info("background job %r of session %s finished (%s); waking it",
                    job_key, session_id, state)
        self.worker.out_q.put({
            "type":       "job_complete",
            "job_key":    job_key,
            "server":     descriptor.get("server"),
            "kind":       descriptor.get("kind"),
            "state":      state,
            "summary":    summary,
            "status_op":  status_op,
            "summary_op": summary_op,
            "session_id": session_id,
            "reason":     reason if state == "unknown" else "",
        })
        # Deliberately not marked as reported here. Emitting is not delivering: the
        # event still has to be folded into a turn by whatever consumes the bus, and a
        # run settled at the moment its event is queued is one every later scan takes
        # for already delivered — so a wake nobody has read is a wake nobody ever will.
        # The consumer writes the marker — see ``jobs_on_disk.mark_wakes_reported``.

    def watched_runs(self) -> list[dict]:
        """The runs held by watchers, for the panel: descriptor and last status, minus the local ``at``."""
        out = []
        for watch in self.watches.values():
            descriptor = watch.descriptor
            if not isinstance(descriptor, dict) or watch.task.done():
                continue
            status = {k: v for k, v in (watch.status or {}).items() if k != "at"}
            out.append({
                "job_key": descriptor.get("job_key", "?"),
                "kind": descriptor.get("kind", ""),
                "server": descriptor.get("server", ""),
                "origin": descriptor.get("origin") or {},
                **status,
            })
        return out


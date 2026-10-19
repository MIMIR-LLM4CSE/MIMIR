"""The runs a conversation launched and left running: watched, reported, checked on.

A watcher polls a run's status op until it ends and puts a ``job_complete`` on the
outbox for the pool's ``WakeRouter``; a check-in cycle sends bulletins meanwhile.
"""
from __future__ import annotations

import asyncio
import logging
from itertools import chain, repeat
from typing import TYPE_CHECKING

from ....session.jobs_on_disk import (
    ended_event,
    needing_attention,
    scan_session,
    subagent_sessions,
)
from ....query_engine.jobs import RunWatcher, Tick, Watch

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


class JobWatcher(RunWatcher):
    """One conversation's background runs and its check-in cycle."""

    def __init__(self, worker: "AgentWorker") -> None:
        super().__init__()
        self.worker = worker
        # The check-in cycle covering every job of this conversation, or None between
        # waves. One per conversation rather than one per job: three jobs launched in
        # the same step share a schedule and report together instead of waking it
        # three times a minute apart.
        self._checkin_task: asyncio.Task | None = None
        # Fingerprint of the last bulletin sent, so one with nothing new is not.
        self._checkin_last: tuple | None = None

    def tasks(self) -> list[asyncio.Task]:
        """Every task this holds — watchers and the check-in cycle — for shutdown."""
        tasks = super().tasks()
        if self._checkin_task is not None:
            tasks.append(self._checkin_task)
        return tasks

    # ── RunWatcher ──────────────────────────────────────────────────────────────

    def tool_runner(self):
        return self.worker.agent.run_tool

    def session_for(self, owner: str | None) -> str | None:
        return owner or self.worker.query_session_id or self.worker.session_id

    def adopt(self, descriptor: dict, session_id: str | None) -> None:
        # Whose run this is, which is not the same question as who gets woken: a run a
        # sub-agent launched and handed over is watched for the parent and still named
        # after the child. setdefault, never an assignment: a descriptor that arrived
        # with an origin arrived with the only end that knew it.
        descriptor.setdefault("origin", {"kind": "main", "label": "", "session": session_id})

    def on_registered(self, job_key: str, descriptor: dict, session_id: str | None) -> None:
        self._start_checkins(session_id)

    def on_tick(self, job_key: str, watch: Watch, tick: Tick, fresh_alerts: dict,
                session_id: str | None) -> None:
        # A run gone wrong is not left waiting for the next point of the schedule.
        if fresh_alerts:
            self._emit_checkin(session_id)
        # Starts at ("", None), so a run that never reports is never sent, while one
        # whose count disappears is — a retraction is news too.
        if (tick.phase, tick.percent) != watch.progress:
            watch.progress = (tick.phase, tick.percent)
            self.worker.out_q.put({
                "type":       "job_progress",
                "job_key":    job_key,
                "phase":      tick.phase,
                "percent":    tick.percent,
                "session_id": session_id,
            })

    def deliver(self, job_key: str, event: dict) -> None:
        # Not marked as reported here: emitting is not delivering, and a run settled
        # when its event is queued is one every later scan takes for delivered. The
        # consumer writes the marker (``jobs_on_disk.mark_wakes_reported``).
        self.worker.out_q.put(event)

    def rearm_detached_jobs(self) -> dict:
        """Put watchers back on this session's runs and report those that ended; return what it did.

        Only runs no watcher holds; a restart loses the promise to report, never the run.
        """
        session_id = self.worker.session_id
        if not session_id:
            return {"rearmed": [], "reported": []}
        # Its sub-agents' runs too: a child killed outright hands nothing over, and no
        # front end can load a sub-session.
        jobs = needing_attention({sid: scan_session(sid)
                                  for sid in (session_id, *subagent_sessions(session_id))})
        rearmed: list[str] = []
        reported: list[str] = []
        for job in jobs:
            if job.job_key in self.watched_job_keys():
                continue   # a watcher is already holding it
            if job.live:
                if self.register(job.descriptor(), owner=session_id):
                    rearmed.append(job.job_key)
                continue
            # Settled by the consumer, not here (``jobs_on_disk.mark_wakes_reported``).
            self.worker.out_q.put(ended_event(job))
            reported.append(job.job_key)
        if rearmed or reported:
            logger.info("job scan: session %s re-armed %d run(s) and reported %d "
                        "that had ended", session_id, len(rearmed), len(reported))
        return {"rearmed": rearmed, "reported": reported}

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


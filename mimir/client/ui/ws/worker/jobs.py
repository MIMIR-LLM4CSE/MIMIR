"""The runs a conversation launched and left running: watched, reported, checked on.

Its ``job_complete`` and ``job_checkin`` go on the outbox for the pool's ``WakeRouter``;
its progress goes to the panel; the runs found on disk after a restart are re-armed.
"""
from __future__ import annotations

import logging
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


class JobWatcher(RunWatcher):
    """One conversation's background runs, delivered to the pool."""

    def __init__(self, worker: "AgentWorker") -> None:
        super().__init__()
        self.worker = worker

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

    def on_tick(self, job_key: str, watch: Watch, tick: Tick, session_id: str | None) -> None:
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

    def deliver(self, event: dict) -> None:
        # The pool's WakeRouter hands it to a turn. Not marked as reported here: emitting is not delivering, and a run settled
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


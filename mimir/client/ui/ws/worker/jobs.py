"""The runs a conversation launched and left running: watched, reported, checked on.

Its ``job_complete`` and ``job_checkin`` go on the outbox for the pool's ``WakeRouter``;
its progress goes to the panel; the runs found on disk after a restart are re-armed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ....context.checkpoints import Checkpoints
from ....session.jobs_on_disk import (
    ended_event,
    mark_job_reported,
    needing_attention,
    scan_session,
    subagent_sessions,
)
from ....query_engine.jobs import RunWatcher, Tick, Watch
from ....tool_execution.direct import call_tool_directly
from .....shared.responses import err

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
        # Started by a turn: a rewind past the message that opened it stops it. A run
        # re-armed after a restart is no turn's, and stays.
        checkpoints = Checkpoints.of(session_id)
        if checkpoints is not None and self.worker.is_busy():
            checkpoints.note_run(descriptor)

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

    async def cancel(self, job_key: str) -> dict:
        """Stop *job_key* with its descriptor's cancel op, then poll it at once; the op's answer.

        The user's own action, so the call goes around the guardrails. The watch is not
        dropped here: the next poll reads the run as ended and wakes the conversation
        like any other ending, which is how the agent learns of it.
        """
        cancel_op = self.cancel_op(job_key)
        if cancel_op is None:
            return err("This run cannot be stopped from here.")
        payload = await call_tool_directly(self.worker.agent, cancel_op["tool"],
                                           dict(cancel_op.get("args") or {}))
        self.poll_now(job_key)
        return payload

    async def withdraw(self, job_key: str) -> dict:
        """Stop *job_key* and drop its watch without waking anyone; the cancel op's answer.

        For a rewind: the turns that started the run are gone, and its ending must not
        come back as a wake of the conversation that no longer holds them.
        """
        cancel_op = self.cancel_op(job_key)
        with self._lock:
            watch = self.watches.pop(job_key, None)
        if watch is not None and not watch.task.done():
            watch.task.cancel()
        server = watch.descriptor.get("server") if watch is not None else None
        mark_job_reported(self.session_for(None), job_key, server)
        if cancel_op is None:
            return err("This run cannot be stopped from here.")
        return await call_tool_directly(self.worker.agent, cancel_op["tool"],
                                        dict(cancel_op.get("args") or {}))

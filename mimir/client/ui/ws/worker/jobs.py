"""The runs a conversation launched and left running: watched, reported, checked on.

A detached run — a build, a training, a Slurm job — outlives the turn that started it.
The watcher polls it (read-only, through the run's own status op) on the worker's
loop until it ends, then puts a ``job_complete`` on the worker's outbox; the pool's
``WakeRouter`` hands that to a turn of the conversation that launched it. While runs
are going, a check-in cycle sends a short bulletin on a widening schedule, so a run
gone wrong is found out early rather than at its end.

Split out of the worker, which kept all of this among its own state.
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


# When a background check-in fires, as seconds from the one before it: T+30s, T+2.5min,
# T+12.5min, T+42.5min, and hourly from there for as long as the run lasts. A detached
# run says nothing between its launch and its end, so a two-hour build that went wrong
# in its third minute is found out in its hundred-and-twentieth.
#
# The ramp widens the way the watcher's own backoff does, and for the same reason: so
# does the cost of having been wrong for that long. Early on, a mistake is minutes of
# cluster time and the gap should be small; four hours in, a bulletin every few minutes
# would be noise charged against a turn each time. The hourly tail is the floor under
# that — an overnight job stays answerable for without ever being chatty, where a
# schedule that simply ran out would leave the longest runs, the ones with most to
# lose, as silent as before.
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
    """A live background-job watcher: the polling task, and what it is watching.

    The descriptor is kept next to the task because the dispatch guard compares an
    incoming call against the *job's own* ``status_op`` — registry data travelling on
    the descriptor, the same reason the watcher itself can poll generically. Holding
    only the task would have forced the guard to name a tool.

    ``status`` is the last thing the poll learned — ``{state, phase, percent, at}``,
    whatever the status op chose to report, shape-driven like everything else here. A
    check-in reads it instead of polling again: the watcher is already asking, once a
    tick, and a second asker would double every run's status traffic to say what the
    first one already knows. It has to be mutable, which is what a dataclass buys here.
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
        self._bg_jobs: dict[str, _Watch] = {}
        # The check-in cycle covering every job of this conversation, or None between
        # waves. One per conversation rather than one per job: three jobs launched in
        # the same step share a schedule and report together instead of waking it
        # three times a minute apart.
        self._checkin_task: asyncio.Task | None = None
        # Fingerprint of the last bulletin sent, so one with nothing new is not.
        self._checkin_last: tuple | None = None

    def tasks(self) -> list[asyncio.Task]:
        """Every task this holds — watchers and the check-in cycle — for shutdown."""
        tasks = [w.task for w in self._bg_jobs.values()]
        if self._checkin_task is not None:
            tasks.append(self._checkin_task)
        return tasks

    def register(self, descriptor: dict, owner: str | None = None) -> bool:
        """Register a completion watcher for a detached run (the agent's hook).

        Called on the worker loop from the agent's tool dispatch. Dedups on
        ``job_key`` so re-launching the same job never spawns a second watcher.
        Returns True when a watcher is (already) active — the loop uses this to
        tell the model it may end its turn.

        The session that launched the job is captured here and travels with the
        watcher: a two-hour build outlives the conversation on screen, and the wake
        belongs to the conversation that asked for it, not to whichever one the user
        happens to be reading when it lands.

        *owner* names that session explicitly. The running query is the right answer
        when the agent registers a job it has just launched, and the wrong one when a
        watcher is being put back on a run found on disk — there is no running query
        then, and the session is whichever directory the descriptor came out of.
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
        existing = self._bg_jobs.get(job_key)
        if existing is not None and not existing.task.done():
            return True  # already watched
        session_id = owner or self.worker._query_session_id or self.worker._own_session()
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
        self._bg_jobs[job_key] = _Watch(task, descriptor)
        logger.info("background job %r of session %s is now watched by %r",
                    job_key, session_id, (descriptor.get("status_op") or {}).get("tool"))
        self._start_checkins(session_id)
        return True

    def rearm_detached_jobs(self) -> dict:
        """Put watchers back on this session's runs, and report the ones that ended.

        What a restart takes away is the *promise to report* a run, never the run: it
        keeps going in its own session directory, indifferent. So this reads the
        descriptors off disk and makes the promise again — a watcher for everything
        still going, and a ``job_complete`` for everything that finished while nothing
        was listening, which is the wake its conversation has been owed since.

        Only runs nothing is already watching. ``_bg_jobs`` is the authority on that,
        and the ordinary case — a server that simply stayed up — finds it full and does
        nothing at all. Returns what it did, for the log and for tests.
        """
        session_id = self.worker._own_session()
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
            existing = self._bg_jobs.get(job.job_key)
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
        """Report a watcher or check-in cycle that ended on an exception.

        Nothing awaits these tasks, so an exception escaping one is held until the GC
        notices and prints "Task exception was never retrieved" — long after the run it
        was holding finished into silence. The failure and the missing wake are the same
        event, and this is the only place they can be named together.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("background-job task died: %r — any run it was holding will "
                           "not wake anyone", exc, exc_info=exc)

    def _start_checkins(self, session_id: str | None) -> None:
        """Begin a check-in cycle for this conversation, unless one is already running.

        Called on every registration, and a no-op for all but the first of a wave: a job
        joining a cycle in flight inherits what is left of it rather than restarting the
        clock, which is what keeps three jobs launched together to one bulletin apiece
        instead of three.

        What decides is whether this registration *starts* a wave — whether it is the
        only live run of the conversation. The cycle cannot end on its own the instant
        the last job does: its tail is hourly, so it is asleep for up to an hour after
        a wave finishes, and a job launched into that gap would inherit the remainder
        of a schedule that no longer describes anything and wait out the hour for its
        first bulletin. A new wave therefore replaces the cycle rather than joining it.
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
        """Emit ``job_checkin`` down ``_CHECKIN_SCHEDULE``, then hourly while runs last.

        Reports what the watchers have already seen — ``_Watch.status``, written by the
        poll that is running anyway — so a check-in costs no status traffic at all. A
        point with nothing new since the last bulletin says nothing.

        Ends the moment this conversation has no live run left: past that the completion
        wakes have landed and there is nothing to be reassured about. That is the only
        thing that ends it — the tail does not run out — so an overnight run is still
        answered for at hour six, where a schedule with a last point would have gone
        quiet exactly where the stakes were highest.
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
            for key, w in self._bg_jobs.items() if not w.task.done()
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
        """Keys of the runs this conversation still has in flight.

        Asked before delivering a check-in it held back — a bulletin on runs that have
        since finished is worse than none — and by the panel, which reports a live run
        beside a live turn so walking away from one is a question rather than a
        surprise.

        """
        return [k for k, w in self._bg_jobs.items() if not w.task.done()]

    def watched(self) -> list[dict]:
        """Descriptors of the runs a watcher is currently holding (the agent's hook).

        The deterministic half of the dispatch guard: a job is in this list or it is
        not, so nothing has to be inferred about what the model meant. Read on the
        worker loop, which is also the only thread that writes ``_bg_jobs``.

        Finished tasks are filtered rather than trusted to have been popped: a watcher
        that raised leaves its entry behind, and a guard that refused calls on a job
        nobody is watching would block the one question the model still needs to ask.
        """
        return [w.descriptor for w in self._bg_jobs.values()
                if isinstance(w.descriptor, dict) and not w.task.done()]

    async def _watch_job(self, job_key: str, descriptor: dict,
                         session_id: str | None = None) -> None:
        """Poll a detached run to completion, then emit ``job_complete`` on ``out_q``.

        An independent task on the worker loop that outlives the launching turn.
        Read-only status polling only (safe to interleave with an active turn on the
        single-threaded loop). The WS session handles the UI notification and the
        auto-resume (``WakeRouter``).

        The poll itself is :func:`poll.poll_until_terminal`, shared with the in-turn
        fallback and with a sub-agent's own watcher; what belongs here is the schedule it
        is given, the check-in and progress it writes per tick, and the event below.

        The event carries the descriptor's own ops. The session that reads it knows
        nothing about what kind of job this was, so what it can say to the model comes
        from the descriptor and the summary rather than from anything it names itself.
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
            self._bg_jobs.pop(job_key, None)

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
            watch = self._bg_jobs.get(job_key)
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
            self.worker._agent.run_tool, status_op,
            interval=5.0, max_interval=30.0,
            call_timeout=poll.STATUS_CALL_TIMEOUT,
            unreadable_limit=poll.UNREADABLE_POLL_LIMIT,
            on_tick=_tick, label=job_key)

        summary = await poll.read_summary(
            self.worker._agent.run_tool, summary_op,
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
        """What is running outside the current turn: detached jobs the watcher holds.

        The descriptor says what kind of run it is and whose it is; the rest is what
        the poll learned and was already keeping — where it is running, how far it has
        got, which pid or job id it wears. Both halves read without being understood,
        so a status op that starts reporting something new needs no change here.

        ``at`` is deliberately dropped: it is a ``time.monotonic()`` reading, which
        means nothing outside this process.
        """
        out = []
        for watch in self._bg_jobs.values():
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


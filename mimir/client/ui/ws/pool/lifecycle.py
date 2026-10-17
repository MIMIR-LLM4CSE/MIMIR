"""When the pool lets an idle worker go, and when the whole server may stop.

A worker idle past its TTL is released (its conversation is on disk); the server
stops once nothing is left that anyone could come back to: no connection, no
worker, no detached run, no claim from a window that asked to keep it."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any
from ..server import registry as server_registry
from ....session.jobs_on_disk import has_baseline, scan_all_sessions
from ..worker.human import detach_grace
from ..worker.worker import AgentWorker

logger = logging.getLogger(__name__)


def server_idle_ttl() -> float:
    try:
        return max(60.0, float(
            os.environ.get("MIMIR_SERVER_IDLE_TTL", "") or _DEFAULT_SERVER_IDLE_TTL))
    except ValueError:
        return _DEFAULT_SERVER_IDLE_TTL


def idle_stop_reason(ttl_secs: float, *, past: bool) -> str:
    """What this TTL means, in whichever tense the sayer needs.

    Two layers describe it to a user — the notice written into a conversation after the
    process stops, and the panel's own line when they detach onto nothing — and they
    were each spelling it out, with the wording and the formatting drifting apart. One
    author with two tenses, so they cannot disagree about what the clock is.

    What neither may say is a deadline. The clock is measured from when the process
    *first* had nothing to do and is reset by any work, so "it stops in two hours" names
    a moment that does not exist; what ends the process is having had nothing to do
    throughout. The remedy is deliberately absent: this layer knows what it was not
    watching, not what kind of work the user has — a scheduler is something only the
    tools that talk to one know about.
    """
    clock = f"{ttl_secs / 3600:.0f}h"
    if past:
        return (f"MIMIR stopped: it had nothing to do for {clock} together — no turn "
                "in flight, and no run it was watching; any work would have restarted "
                "that clock.")
    return (f"It stops only once it has had nothing to do for about {clock} together "
            "— any work restarts that clock.")


# How often the reaper looks. Nothing here is urgent — a worker that lives thirty seconds
# past its welcome costs nothing — and a tight tick would walk the pool for no reason.
_REAP_INTERVAL = 30.0


# How long a worker may sit unused before it is released. Long enough that switching back
# and forth does not pay for a rebuild, short enough that a forgotten conversation stops
# holding twenty subprocesses.
_DEFAULT_IDLE_TTL = 600.0


# How long everything must stay quiet before a detached server stops on its own. Long
# enough that closing a laptop lid for a meeting does not end a run, short enough that a
# forgotten workspace does not hold sixty interpreters overnight.
_DEFAULT_SERVER_IDLE_TTL = 7200.0


def default_idle_ttl() -> float:
    try:
        return max(30.0, float(os.environ.get("MIMIR_SESSION_IDLE_TTL", "") or _DEFAULT_IDLE_TTL))
    except ValueError:
        return _DEFAULT_IDLE_TTL


class PoolLifecycle:
    # ── Is this process still needed ──────────────────────────────────────────

    def idle_report(self, *, excluding: Any = None) -> dict:
        """Why this process is or is not still needed, in one interrogable answer.

        **The criterion is positive.** "Not busy" is not "has finished": a worker is
        also not busy between queries, after a turn that broke, and while it waits
        behind a card. An idle test built on the absence of noise would stop a server
        whose turn merely paused, so a conversation counts as finished only when it has
        actually *concluded* — delivered a final answer (or ended in an error, which is
        also an ending) — which the bus records as it hands that event to the committer.

        And a run launched but not collected is work in progress even with no turn
        running: that is precisely the "I submitted a two-hour build and left" case
        this whole path exists for, so a live job on disk holds the process open — and
        so does one that has *ended* without its wake reaching a turn, which is the same
        debt one step later.

        The rest are not activity but prohibitions: somebody is attached, a card is
        parked, a deferral is owed, a turn is queued for a slot.

        One place, deliberately: the idle shutdown reads it, and so does the question
        "has this window anything worth leaving running?".

        *excluding* is a subscription to leave out of the count, which is what asks
        that second question properly. A window asking what it would leave behind is
        itself an attachment, so counting it answers "yes, you" every time — which is
        how a detachment came to promise that a run keeps working when there was no run
        and the process was about to stop on idle. Only the asker, not every client: a
        second window of the same workspace *is* keeping this process alive, and
        telling the first that nothing is watching would be the same mistake inverted.
        """
        reasons: list[str] = []
        attached = self.bus.attached(excluding=excluding)
        if attached:
            reasons.append(f"{attached} client(s) attached")
        if self._queue:
            reasons.append(f"{len(self._queue)} turn(s) waiting for a slot")
        if self._building:
            reasons.append(f"{len(self._building)} agent(s) being built")

        unfinished: list[str] = []
        for session_id, worker in list(self._workers.items()):
            try:
                if worker.has_work_pending():
                    reasons.append(f"session {session_id} has a turn in flight")
                    continue
            except Exception:
                reasons.append(f"session {session_id} could not be asked")
                continue
            if worker.human.awaiting_user():
                reasons.append(f"session {session_id} is parked on a card")
                continue
            if worker.human.has_deferral:
                reasons.append(f"session {session_id} is owed an answer")
                continue
            if self.bus.concluded_at(session_id) is None:
                # Never answered, and not running: a turn that ended without saying so.
                # Counted as working, because nothing observed it finishing.
                unfinished.append(session_id)
        if unfinished:
            reasons.append("session(s) that never delivered an answer: "
                           + ", ".join(sorted(unfinished)))

        live_jobs, owed_wakes = self._job_keys()
        if live_jobs:
            reasons.append("background run(s) still going: " + ", ".join(live_jobs))
        if owed_wakes:
            reasons.append("run(s) whose wake nobody has taken in yet: "
                           + ", ".join(owed_wakes))

        return {"idle": not reasons, "reasons": reasons}

    @staticmethod
    def _job_keys() -> tuple[list[str], list[str]]:
        """The runs still going, and the finished ones still owed a wake.

        Read from disk rather than from the live workers: a run outlives the agent that
        launched it, and the session it belongs to may have no worker at all right now.
        One scan answers both, since ``scan_all_sessions`` already drops the runs whose
        wake has been delivered — what it returns is exactly "still going" plus "ended
        and unspoken for".

        **Why an owed wake holds the process open.** A finished run that nothing has
        taken in is a turn this process still has to start, which is work in progress
        as surely as the run itself was. Stopping here is how a conversation loses the
        night: the job ends, its wake is journaled, and the server that owes the turn
        shuts down two hours later without ever having started it.

        Only sessions that have been scanned before are counted. A job directory is
        never swept, however old, so an unbaselined session holds every run it ever
        finished with no marker on any of them — reading that history as debt would
        leave a workspace permanently un-stoppable, which is worse than stopping one
        with a wake outstanding and exactly the trade the baseline exists to make.
        """
        live: list[str] = []
        owed: list[str] = []
        try:
            for session_id, jobs in scan_all_sessions().items():
                baselined = has_baseline(session_id)
                for job in jobs:
                    if job.live:
                        live.append(f"{session_id}/{job.job_key}")
                    elif baselined:
                        owed.append(f"{session_id}/{job.job_key}")
        except Exception:
            # Unreadable means unknown, and unknown must not read as "nothing is
            # running": erring the other way stops a server mid-build.
            logger.warning("pool: the detached runs could not be read", exc_info=True)
            return ["<unreadable>"], []
        return sorted(live), sorted(owed)

    def request_stop(self, headline: str = "") -> None:
        """Ask the serve loop to shut down. Idempotent.

        *headline* is why, when the answer is this process's own: it is recorded into
        every conversation being kept before the loop is asked to go. Recorded here
        rather than at each deciding branch so that "the process decided to stop" is
        one fact in one place — a stop added later cannot forget to say so. The
        user-requested stop passes nothing, and should: the user is there, and already
        has the reasons on screen.
        """
        event = self.stop_requested
        if event is not None and not event.is_set():
            if headline:
                self._record_own_stop(headline)
            event.set()

    # ── Releasing ─────────────────────────────────────────────────────────────

    def releasable(self, session_id: str) -> bool:
        """Whether this worker may be closed without throwing work or attention away.

        Each clause is a thing that would be lost:

        * **busy** — a turn is running, or is queued and about to.
        * **watching a background job** — a two-hour build's watcher lives on this
          worker's loop. Evicting it loses the wake that watcher exists to deliver, so
          the build finishes and nothing ever says so.
        * **parked on a card** — the turn is waiting on a person, with no timeout, by
          design. Closing it discards a question the user may be about to answer.
        * **holding a deferral** — the same debt, set aside. A deferred turn has
          cleared its pending card, so the clause above no longer sees it, and the
          user still owes it an answer; releasing the agent here throws away the ~19
          servers that answer will be resumed against.
        * **owed a wake** — a run of its finished and no turn has taken that news in
          yet. The debt one step later than "watching a job", and the clause that
          closes the gap between them: a watcher drops the job the instant it reports
          it, so without this the agent is releasable for the one moment at which it
          is needed most — and the conversation it belongs to has by definition been
          idle for hours, because it was waiting on a long run.
        * **holding undrained output** — its last word has not been acted on. Each
          event wakes the pump, which does the whole handover inside one synchronous
          tick; before it, closing the worker discards what it had just said — the
          completion event included — and the agent that event is addressed to with
          it.
        * **on screen** — the conversation the user is reading must stay instant.

    Together these are what make "ten minutes idle" safe to act on. Idleness is counted
    from the last time the pool was asked for the worker, which for a conversation
    waiting on an overnight run is hours ago throughout — so the clock never protects
    it and these clauses are the whole of what does. Their sum is one invariant: an
    agent is released only when nothing of its is unresolved, which is why a wake can
    never arrive for a conversation whose agent has been closed.
        """
        worker = self._workers.get(session_id)
        if worker is None:
            return False
        if session_id == self.active_session_id:
            return False
        try:
            if worker.has_work_pending():
                return False
        except Exception:
            logger.debug("releasable: a best-effort step failed and was skipped", exc_info=True)
            return False
        if worker.watched_job_keys():
            return False
        if worker.human.awaiting_user():
            return False
        if worker.human.has_deferral:
            return False
        conversation = self.conversations._open.get(session_id)
        if conversation is not None and conversation.pending_wakes:
            return False
        out_q = getattr(worker, "out_q", None)
        if out_q is not None:
            try:
                if not out_q.empty():
                    return False
            except Exception:
                logger.debug("releasable: a best-effort step failed and was skipped", exc_info=True)
                return False
        return True

    def idle_for(self, session_id: str) -> float:
        return time.monotonic() - self._last_use.get(session_id, 0.0)

    def _take_one_idle_locked(self) -> AgentWorker | None:
        """Take out the longest-idle releasable worker, to free its slot. Caller holds
        the lock, and closes what it is given once it has let go of it.

        None when nothing may be released — every worker is working, waiting on the
        user, or being read. The caller then queues instead of evicting: taking a slot
        from a conversation that is mid-task to give it to a new one trades a visible
        wait for silent lost work.
        """
        candidates = sorted(
            (sid for sid in self._workers if self.releasable(sid)),
            key=self.idle_for, reverse=True,
        )
        if not candidates:
            return None
        return self._take_locked(candidates[0], reason="a slot was needed")

    async def release_idle(self) -> list[str]:
        """Close every releasable worker idle past the TTL. Returns the sessions closed."""
        closed: list[str] = []
        taken: list[AgentWorker] = []
        async with self._lock:
            for session_id in list(self._workers):
                if not self.releasable(session_id):
                    continue
                if self.idle_for(session_id) < self.idle_ttl:
                    continue
                taken.append(self._take_locked(session_id, reason="idle"))
                closed.append(session_id)
        await self._close_workers(taken)
        return closed

    def ensure_reaper(self) -> None:
        """Start the idle sweep, if it is not already running."""
        if self._reaper is not None and not self._reaper.done():
            return
        try:
            self._reaper = asyncio.get_running_loop().create_task(self._reap_loop())
        except RuntimeError:
            self._reaper = None   # no loop (tests calling the pool synchronously)

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(_REAP_INTERVAL)
            try:
                await self.release_idle()
                await self.pump()
                self._consider_stopping()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("pool: idle sweep failed", exc_info=True)

    def _nobody_asked_to_keep_it(self) -> bool:
        """Whether this process has lost its last window with no claim on it.

        A server exists in one of two states and there is no third: claimed by at least
        one conversation — detached, which redirected its output to a log and asked not
        to be killed — or owned by the window that started it, dying with it. What used
        to be possible was a server in neither: one a window had merely attached to,
        nobody's to kill, still writing to a pipe whose reader is gone. It kept working,
        indifferently and unobserved, and nobody had asked it to.

        So a server nobody claimed stops when its last client has been gone for the
        grace. Server-side, because the only place that can decide this reliably is the
        process itself: an extension host that is dying has no time to end a process it
        does not own, and may be killed before it tries. The grace is the one a parked
        card already uses — a window reload closes and reopens the socket, and that is
        not a departure.

        This is deliberately not conditioned on what is in flight. A run still going
        does not earn a server the right to survive unasked: the panel asks before the
        window goes, and "keep them going" is what claims it. The run itself is not
        killed either way — it has its own process session and an exit code trap, and
        the next agent built for its conversation picks it up again.
        """
        if self.bus is None or not self.bus.ever_attached:
            return False
        unattended = self.bus.unattended_for()
        if unattended is None or unattended < detach_grace():
            return False
        try:
            return not server_registry.claims()
        except Exception:
            # Unreadable means unknown, and unknown must not read as "nobody asked":
            # erring that way stops a detached run because a file could not be read.
            logger.warning("pool: the detach claims could not be read", exc_info=True)
            return False

    def _consider_stopping(self) -> None:
        """Stop once nothing has needed this process for the whole TTL.

        Measured from when it *first* looked idle, and reset by any activity, so the
        answer is "idle throughout" rather than "idle at some point" — which is the
        difference between stopping a forgotten server and stopping one between two
        turns of a conversation the user is coming back to.

        Ahead of all of that: a server nobody claimed and nobody is watching has no
        reason to live at all, whatever it is in the middle of. See
        :meth:`_nobody_asked_to_keep_it`.
        """
        if self.stop_requested is None:
            return
        if self._nobody_asked_to_keep_it():
            logger.info("pool: no window and no conversation asking to be left "
                        "running — stopping")
            self.request_stop(
                "MIMIR stopped: its last window closed and no conversation had "
                "asked for the run to be left going.")
            return
        report = self.idle_report()
        if not report["idle"]:
            if self._idle_since is not None:
                logger.info("pool: no longer idle (%s)", "; ".join(report["reasons"]))
            self._idle_since = None
            return
        now = time.monotonic()
        if self._idle_since is None:
            self._idle_since = now
            logger.info("pool: nothing needs this process; stopping in %.0fs unless "
                        "something does", self.server_idle_ttl)
            return
        if now - self._idle_since >= self.server_idle_ttl:
            logger.info("pool: idle for %.0fs — stopping", now - self._idle_since)
            self.request_stop(idle_stop_reason(self.server_idle_ttl, past=True))

    def _record_own_stop(self, headline: str) -> None:
        """Write into every kept conversation that this process stopped itself, and why.

        The counterpart of a run's wake, for the one thing that happens to a
        conversation which the conversation can never see happen. Nothing reached the
        transcript when a server stopped on its own: the decision was printed in the
        process's own log, which the user has no reason to be reading, so a morning
        opened on a thread whose last line was hours old with nothing between — a night
        that reads as lost rather than as a decision taken and recorded.

        Stamped with the wall-clock time, which is the half of the answer the file
        itself cannot give: the gap in the thread is what raises the question, and
        "stopped at 00:38" is what closes it.

        Best effort throughout, and on purpose: this runs on the way out, and a
        conversation whose file will not load is not worth holding a shutdown for.
        """
        stamped = f"⏹ {datetime.now().astimezone():%Y-%m-%d %H:%M} — {headline}"
        try:
            kept = sorted(server_registry.claims())
        except Exception:
            logger.warning("pool: the detach claims could not be read for the stop "
                           "notice", exc_info=True)
            kept = []
        # The conversations that were being kept, or — in the unclaimed case, where by
        # definition there are none — the ones with an agent still here. Those are the
        # threads a user comes back to.
        for session_id in kept or sorted(sid for sid, _w in self.items()):
            conversation = self.conversations.get(session_id)
            if conversation is None:
                continue
            conversation.note(stamped)
            if self.save_conversation(conversation):
                self.bus.record_client_event(
                    session_id, {"type": "server_stopped", "text": headline})

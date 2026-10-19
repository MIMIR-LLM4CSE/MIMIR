"""When the whole server may stop, and how long a worker may sit idle.

The server stops once nothing is left that anyone could come back to: no connection,
no worker, no detached run, no claim from a window that asked to keep it."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from ..process import registry as server_registry
from ....session.jobs_on_disk import has_baseline, scan_all_sessions
from ..worker.human import detach_grace

if TYPE_CHECKING:
    from .pool import AgentPool

logger = logging.getLogger(__name__)


def server_idle_ttl() -> float:
    try:
        return max(60.0, float(
            os.environ.get("MIMIR_SERVER_IDLE_TTL", "") or _DEFAULT_SERVER_IDLE_TTL))
    except ValueError:
        return _DEFAULT_SERVER_IDLE_TTL


def idle_stop_reason(ttl_secs: float, *, past: bool) -> str:
    """What the server TTL means, in the past or present tense, for the user.

    Never a deadline: the clock starts when the process first had nothing to do, and any
    work resets it.
    """
    clock = f"{ttl_secs / 3600:.0f}h"
    if past:
        return (f"MIMIR stopped: it had nothing to do for {clock} together — no turn "
                "in flight, and no run it was watching; any work would have restarted "
                "that clock.")
    return (f"It stops only once it has had nothing to do for about {clock} together "
            "— any work restarts that clock.")


# How often the pool's reaper looks. Nothing here is urgent — a worker that lives thirty seconds
# past its welcome costs nothing — and a tight tick would walk the pool for no reason.
REAP_INTERVAL = 30.0


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


class Lifecycle:
    """Whether this process is still needed, and stopping it once it is not.

    Reads the pool it belongs to through its public face: the bus, the workers, the
    waiting line, the conversations and their writer.
    """

    def __init__(self, pool: AgentPool) -> None:
        self._pool = pool
        # Resolved when this process decides it is no longer needed. Awaited beside the
        # signal handlers, so an idle stop and a SIGTERM take the same path out — the
        # one that closes the MCP servers.
        self.stop_requested: asyncio.Event | None = None
        self.server_idle_ttl = server_idle_ttl()
        # When the whole process first looked idle, or None while it does not. The TTL
        # is measured from here, so a moment of activity restarts it rather than
        # shortening it.
        self._idle_since: float | None = None

    def idle_report(self, *, excluding: Any = None) -> dict:
        """``{"idle", "reasons"}``: why this process is or is not still needed.

        Positive criterion: a conversation is finished only once it concluded. Live or owed
        runs, attachments, parked cards, deferrals and queued turns hold it. *excluding* leaves
        the asking subscription out of the count.
        """
        reasons: list[str] = []
        attached = self._pool.bus.attached(excluding=excluding)
        if attached:
            reasons.append(f"{attached} client(s) attached")
        waiting, building = self._pool.waiting(), self._pool.building()
        if waiting:
            reasons.append(f"{waiting} turn(s) waiting for a slot")
        if building:
            reasons.append(f"{building} agent(s) being built")

        unfinished: list[str] = []
        for session_id, worker in self._pool.items():
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
            if self._pool.bus.concluded_at(session_id) is None:
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
        """``(live, owed)``: runs still going, and finished ones whose wake no turn took, from disk.

        Only baselined sessions count, or an old workspace would never stop.
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
        """Ask the serve loop to stop (idempotent), recording *headline* into kept conversations."""
        event = self.stop_requested
        if event is not None and not event.is_set():
            if headline:
                self._record_own_stop(headline)
            event.set()

    def _nobody_asked_to_keep_it(self) -> bool:
        """Whether the last client has been gone past the grace with no detach claim.

        A server is either claimed or owned by its window; an unclaimed one stops, whatever
        runs (the runs survive on their own).
        """
        if self._pool.bus is None or not self._pool.bus.ever_attached:
            return False
        unattended = self._pool.bus.unattended_for()
        if unattended is None or unattended < detach_grace():
            return False
        try:
            return not server_registry.claims()
        except Exception:
            # Unreadable means unknown, and unknown must not read as "nobody asked":
            # erring that way stops a detached run because a file could not be read.
            logger.warning("pool: the detach claims could not be read", exc_info=True)
            return False

    def consider_stopping(self) -> None:
        """Stop when unclaimed and unwatched, or when idle throughout the server TTL."""
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
        """Note in every kept conversation, with the time, that this process stopped itself and why."""
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
        for session_id in kept or sorted(sid for sid, _w in self._pool.items()):
            conversation = self._pool.conversations.get(session_id)
            if conversation is None:
                continue
            conversation.note(stamped)
            if self._pool.writer.save_conversation(conversation):
                self._pool.bus.record_client_event(
                    session_id, {"type": "server_stopped", "text": headline})

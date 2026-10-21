"""How a conversation is told what happened while it was not looking.

Finished runs, bulletins and unread steering each become a steer into the running
turn or a turn of their own. Synchronous, run by the pump before any socket sees them;
a conversation with no agent is queued for one. Runs that ended with nothing watching
are found on disk by :meth:`WakeRouter.sweep`, on every connection and every reap.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ....protocol.events import DURABLE_EVENTS
from ....session.conversation import Conversation
from ....session.jobs_on_disk import (
    ended_event,
    mark_wakes_reported,
    needing_attention,
    scan_all_sessions,
    wake_reported,
)
from ....query_engine.jobs.wakes import checkin_text, wake_text
from ..worker.worker import AgentWorker

if TYPE_CHECKING:
    from .pool import AgentPool

logger = logging.getLogger(__name__)


class WakeRouter:
    """Hands each finished run, bulletin and unread steer to a turn of its conversation."""

    def __init__(self, pool: "AgentPool") -> None:
        self.pool = pool

    # ── A run finished, or is still going ─────────────────────────────────────

    def consume(self, ev: dict) -> bool:
        """Route a ``job_complete`` or ``job_checkin``. True when it started a turn.

        Never raises: the pump serves every conversation, and one unreadable session
        must not stop the tick draining all the others.
        """
        etype = ev.get("type")
        owner = ev.get("session_id")
        if not owner or etype not in DURABLE_EVENTS:
            return False
        conversation = self.pool.conversations.get(owner)
        if conversation is None:
            logger.warning("%s for job %r arrived for session %s, which does not exist",
                           etype, ev.get("job_key"), owner)
            return False
        worker = self.pool.get(owner)
        if etype == "job_checkin":
            return self._checkin(conversation, worker, ev)
        return self._complete(conversation, worker, ev)

    def _complete(self, conversation: Conversation, worker: "AgentWorker | None",
                  ev: dict) -> bool:
        """Hand a finished run to the conversation that launched it: steered into its running turn, else a new turn."""
        job_key = ev.get("job_key")
        # One wake per run, however many times it is announced: the watcher holding the
        # run and a scan that found it unspoken-for can both speak before either
        # delivers, and the marker that settles it is written at delivery.
        if wake_reported(ev):
            logger.info("wake for job %r of session %s already delivered; dropping the "
                        "re-announcement", job_key, conversation.id)
            return False
        if job_key and any(i["ev"].get("job_key") == job_key
                           for i in conversation.pending_wakes):
            logger.info("wake for job %r of session %s already pending; dropping the "
                        "duplicate announcement", job_key, conversation.id)
            return False
        item = {"ev": ev, "told": False}
        conversation.pending_wakes.append(item)
        if worker is None:
            # An agent is never released while a wake of its is owed, so this is a
            # process that started after the run's watcher died with the last one.
            self._summon(conversation)
            return False
        if self._occupied(conversation.id):
            self._record_wakes(conversation, [item])
            # Left pending on purpose: a steer is only known to have been read when the
            # loop says so (``steer_injected``), and until then the run is still owed a
            # turn of its own.
            worker.submit_steer(wake_text(ev))
            return False
        return self.flush(conversation, worker)

    def _checkin(self, conversation: Conversation, worker: "AgentWorker | None",
                 ev: dict) -> bool:
        """Report runs still going without interrupting; held while busy, latest wins. True if a turn began."""
        if worker is None:
            # Nothing to preserve either: the run is still going and its own wake is
            # still to come.
            logger.warning("check-in for session %s dropped: it has no agent",
                           conversation.id)
            return False
        if self._occupied(conversation.id):
            conversation.held_checkin = ev
            return False
        return self._deliver_checkin(conversation, worker, ev)

    def _deliver_checkin(self, conversation: Conversation, worker: "AgentWorker",
                         ev: dict) -> bool:
        conversation.held_checkin = None
        text = checkin_text(ev)
        if not self.pool.writer.submit_turn(conversation, worker, text):
            return False
        # No bubble: the instruction is addressed to the model, and a conversation
        # checked on twenty times would reopen on twenty blocks of it. The journal still
        # records it — the count of ``job_wake`` lines is how the mechanism is audited,
        # which is why a bulletin is filed under its own type.
        self.pool.bus.record_client_event(
            conversation.id, {"type": "job_checkin", "text": text, "job": ev.get("job_key")})
        return True

    def _summon(self, conversation: Conversation) -> None:
        """Get this conversation an agent, and hand it what it is owed once it has one.

        Through the waiting line, so the cap holds. A turn of its already waiting there
        carries the wakes on landing (``after_turn``).
        """
        if self.pool.queued_position(conversation.id) is not None:
            logger.info("wake for session %s waits on the turn already queued for it",
                        conversation.id)
            return
        logger.info("wake for session %s: building it an agent to run it", conversation.id)
        self.pool.enqueue(conversation.id, lambda worker: self.flush(conversation, worker))
        self.pool.admit_soon()

    # ── Runs nothing watches ──────────────────────────────────────────────────

    def sweep(self) -> None:
        """Wake the conversation of every run that ended on disk with no watcher holding it.

        A run a live watcher holds is left to it: its wake carries the run's summary.
        A run still going is left alone too, and swept up here once it ends: an agent
        built only to watch would hold a slot for as long as the run lasts. Never raises.
        """
        try:
            found = scan_all_sessions()
        except Exception:
            logger.warning("job sweep: the sessions could not be read", exc_info=True)
            return
        held: set[str] = set()
        for _sid, worker in self.pool.items():
            try:
                held.update(worker.watched_job_keys())
            except Exception:
                logger.debug("sweep: a best-effort step failed and was skipped", exc_info=True)
        for job in needing_attention(found):
            if not job.live and job.job_key not in held:
                self.consume(ended_event(job))

    # ── Delivering what is owed ───────────────────────────────────────────────

    def flush(self, conversation: Conversation, worker: "AgentWorker") -> bool:
        """Start one turn carrying every run this conversation is owed, each wake text in full."""
        items, conversation.pending_wakes = conversation.pending_wakes, []
        if not items:
            return False
        self._record_wakes(conversation, [i for i in items if not i["told"]])
        events = [i["ev"] for i in items]
        text = "\n\n".join(wake_text(e) for e in events)
        if not self.pool.writer.submit_turn(conversation, worker, text):
            # Nothing started, so nothing is settled: put them back for the next
            # announcement, answer or scan.
            conversation.pending_wakes = items + conversation.pending_wakes
            return False
        mark_wakes_reported(events)
        logger.info("session %s woken by %s", conversation.id,
                    ", ".join(str(e.get("job_key")) for e in events))
        return True

    def _record_wakes(self, conversation: Conversation, items: list[dict]) -> None:
        """Show each run's wake once, and journal it — whichever route carries it.

        Deliberately not the history: which message a turn is *given* is the flush's
        business, and a run steered into a running turn arrives in that turn's own
        messages instead.
        """
        if not items:
            return
        for item in items:
            conversation.note(f"🔔 {wake_text(item['ev'])}")
            item["told"] = True
        self.pool.writer.save_conversation(conversation)
        for item in items:
            self.pool.bus.record_client_event(conversation.id, {
                "type": "job_wake", "text": wake_text(item["ev"]),
                "job": item["ev"].get("job_key")})

    def steered(self, owner: str | None, text: str) -> None:
        """The running turn read *text*: settle the runs it names, in memory and on disk."""
        conversation = self.pool.conversations.get(owner)
        if conversation is None or not conversation.pending_wakes:
            return
        read: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        for item in conversation.pending_wakes:
            key = str(item["ev"].get("job_key") or "\x00")
            (read if key in text else kept).append(item)
        conversation.pending_wakes = kept
        if read:
            mark_wakes_reported([i["ev"] for i in read])

    # ── A turn ended ──────────────────────────────────────────────────────────

    def after_turn(self, conversation: Conversation, extras: dict) -> list[str]:
        """Carry on after an answer: unread steering, then finished runs, then the held bulletin.

        Returns the steering that was given a turn.
        """
        unconsumed = [t for t in (extras.get("_unconsumed_steer") or []) if t]
        # Wakes share the steer queue, so what the loop left unread can be a run's
        # wake as well as the user's words. Those are still pending and go out with
        # the other runs; only the user's own get the catch-up turn.
        wake_texts = {wake_text(i["ev"]) for i in conversation.pending_wakes}
        steering = [t for t in unconsumed if t not in wake_texts]
        conversation.take_unread_steer(steering)
        # The runs a steer told this turn about, and which the loop did read: settled
        # here, the one moment that is knowable, or the next scan re-announces them.
        unread = set(unconsumed)
        read = [i for i in conversation.pending_wakes
                if i["told"] and wake_text(i["ev"]) not in unread]
        if read:
            conversation.pending_wakes = [i for i in conversation.pending_wakes
                                          if i not in read]
            mark_wakes_reported([i["ev"] for i in read])

        worker = self.pool.get(conversation.id)
        if worker is None:
            if steering or conversation.pending_wakes:
                logger.warning("session %s has unread steering or owed runs and no "
                               "agent to answer them", conversation.id)
            return []
        started_steer: list[str] = []
        if steering and self.pool.writer.submit_turn(conversation, worker, "\n\n".join(steering)):
            started_steer = steering
        started_wakes = self.flush(conversation, worker)
        if started_steer or started_wakes:
            return started_steer
        held = conversation.held_checkin
        if held is not None:
            # Dropped rather than delivered when the runs it described have since
            # finished: their completion wakes say everything it would, and better.
            try:
                still_running = bool(worker.watched_job_keys())
            except Exception:
                still_running = False
            if not still_running:
                conversation.held_checkin = None
            elif not self._occupied(conversation.id):
                self._deliver_checkin(conversation, worker, held)
        return started_steer

    def _occupied(self, session_id: str) -> bool:
        """Running a turn, or parked on a card a person has to answer."""
        return self.pool.is_busy(session_id) or self.pool.is_parked(session_id)

"""The cards a turn parks on, their answers, and the review of the files it wrote.

A handler group of ``Connection``, built with its view state, its socket, the pool
and the shared ``Screen``. ``MESSAGES`` names the inbound message types it answers.
"""
from __future__ import annotations

import logging
import os

from .....query_engine.deferral import KIND_CALLS, take_deferred_calls
from ...worker.worker import AgentWorker
from .base import Handlers

logger = logging.getLogger(__name__)


class CardHandlers(Handlers):
    MESSAGES: dict[str, str] = {
        "approval_response": "on_approval_response",
        "user_question_response": "on_user_question_response",
        "batch_review_accept": "on_batch_review_accept",
        "batch_review_revert": "on_batch_review_revert",
        "batch_review_accept_file": "on_batch_review_accept_file",
        "batch_review_revert_file": "on_batch_review_revert_file",
    }

    async def resend_parked_prompt(self) -> None:
        """Put back every card a turn is parked on, in every conversation."""
        for _sid, worker in self.pool.items():
            prompt = worker.human.pending_prompt()
            if not prompt:
                continue
            try:
                await self.out.send(prompt)
            except Exception:
                logger.debug("_resend_parked_prompt: the client went away mid-send", exc_info=True)
                return

    def _worker_for_answer(self, msg: dict) -> AgentWorker | None:
        """The agent of the conversation the answered card names, or None (then dropped).

        Never the one on screen by default: that would approve a call the user never saw.
        """
        session_id = (msg.get("session_id") or "").strip()
        if not session_id:
            logger.warning("dropping an answer that names no conversation: %s",
                           msg.get("type"))
            return None
        worker = self.pool.get(session_id)
        if worker is None:
            logger.warning("dropping an answer for %s, which has no agent", session_id)
        return worker

    async def on_approval_response(self, msg: dict) -> None:
        answer = {"choice": msg.get("choice", "n"), "approved_files": msg.get("approved_files")}
        if await self._answer_deferred(msg, answer):
            return
        worker = self._worker_for_answer(msg)
        if worker is not None:
            worker.human.resolve_approval(answer["choice"], answer["approved_files"],
                                    str(msg.get("id") or ""))

    async def on_user_question_response(self, msg: dict) -> None:
        if await self._answer_deferred(msg, {"answers": msg.get("answers") or []}):
            return
        worker = self._worker_for_answer(msg)
        if worker is not None:
            # With the card's id: an answer reaches the card it was shown on, and one
            # naming a card that is gone settles nothing.
            worker.human.resolve_question(msg.get("answers"), str(msg.get("id") or ""))

    async def _answer_deferred(self, msg: dict, answer: dict) -> bool:
        """Route an answer to a deferred card into a resume turn. True when handled.

        An answer to a card the user moved past is swallowed: handed to the worker, it
        would sit on the queue and settle the next prompt the user never saw.
        """
        msg_id = msg.get("id")
        if msg_id and msg_id in self.view.stale_prompt_ids:
            return True
        conv = self.screen.conv
        record = conv.pending_interaction
        if not record or not msg_id or msg_id != (record.get("prompt") or {}).get("id"):
            return False
        conv.pending_interaction = None
        # Answered once: a second answer to the same card (a copy of it on screen)
        # must not settle the next live prompt either.
        self.view.stale_prompt_ids.add(msg_id)
        if record.get("kind") == KIND_CALLS:
            # The placeholders leave the record here as they leave the working copy
            # in the loop, so the real results land where they stood.
            take_deferred_calls(conv.history_full, record.get("call_ids") or [])
            take_deferred_calls(conv.history, record.get("call_ids") or [])
        self.screen.save()
        conv.submitted_len = len(conv.history)
        self.screen.worker.submit_resume(record, answer, list(conv.history),
                                  session_id=self.view.active_session_id)
        return True

    async def on_batch_review_accept(self, msg: dict) -> None:
        # User accepted all pending file edits — keep them on disk, clear snapshots.
        if self.screen.worker.agent is not None:
            self.screen.worker.agent.approvals.batch.snapshots.clear()
            self.screen.worker.agent.approvals.batch.forget_reviewed()
        await self.out.send({"type": "batch_status", "files": []})

    async def on_batch_review_revert(self, msg: dict) -> None:
        # User wants to undo all pending file edits — restore originals.
        moved: list[str] = []
        if self.screen.worker.agent is not None:
            approvals = self.screen.worker.agent.approvals
            for path, original in list(approvals.batch.snapshots.items()):
                if not self._revert_one(path, original):
                    moved.append(os.path.relpath(path))
            approvals.batch.snapshots.clear()
            approvals.batch.forget_reviewed()
        await self.out.send({"type": "batch_status", "files": []})
        await self._report_unreverted(moved)

    def _revert_one(self, path: str, original: str | None) -> bool:
        """Restore *path* to *original*; False when it changed since it was reviewed (left alone)."""
        approvals = self.screen.worker.agent.approvals
        if approvals.batch.reviewed_matches(path) is False:
            return False
        abs_path = os.path.abspath(path)
        try:
            if original is None:
                os.remove(abs_path)
            else:
                with open(abs_path, "w", encoding="utf-8") as fh:
                    fh.write(original)
        except OSError:
            pass
        return True

    async def _report_unreverted(self, moved: list[str]) -> None:
        """Say which files were left as they are, and why — never silently."""
        if not moved:
            return
        try:
            await self.out.send({
                "type": "status",
                "text": ("  ⚠ Left unchanged — changed since you were shown the diff, so "
                         "reverting would discard that too: " + ", ".join(sorted(moved))),
            })
        except Exception:
            logger.debug("_report_unreverted: the client went away mid-send", exc_info=True)

    async def on_batch_review_accept_file(self, msg: dict) -> None:
        # Accept a single file — remove its snapshot, keep file on disk.
        file_rel = msg.get("file", "")
        if file_rel and self.screen.worker.agent is not None:
            target = os.path.normpath(os.path.abspath(file_rel))
            snapshots = self.screen.worker.agent.approvals.batch.snapshots
            key = next(
                (k for k in snapshots if os.path.normpath(os.path.abspath(k)) == target),
                None,
            )
            if key is not None:
                del snapshots[key]
                self.screen.worker.agent.approvals.batch.forget_reviewed(key)
        self.screen.worker.review.push_batch_status()

    async def on_batch_review_revert_file(self, msg: dict) -> None:
        # Revert a single file — restore its original content.
        file_rel = msg.get("file", "")
        if file_rel and self.screen.worker.agent is not None:
            target = os.path.normpath(os.path.abspath(file_rel))
            snapshots = self.screen.worker.agent.approvals.batch.snapshots
            key = next(
                (k for k in snapshots if os.path.normpath(os.path.abspath(k)) == target),
                None,
            )
            if key is not None:
                original = snapshots.pop(key)
                reverted = self._revert_one(key, original)
                self.screen.worker.agent.approvals.batch.forget_reviewed(key)
                if not reverted:
                    await self._report_unreverted([os.path.relpath(key)])
        self.screen.worker.review.push_batch_status()

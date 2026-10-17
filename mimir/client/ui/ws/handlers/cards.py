"""The cards a turn parks on, their answers, and the review of the files it wrote.

A part of ``_Session`` (``ws_session``): a mixin over the connection's own state, so a
handler reads and writes the session it serves. ``MESSAGES`` names the inbound message
types it answers.
"""
from __future__ import annotations

import json
import logging
import os

from ....query_engine.deferral import KIND_CALLS, take_deferred_calls
from ..ws_worker import _AgentWorker

logger = logging.getLogger(__name__)


class CardHandlers:
    MESSAGES: dict[str, str] = {
        "approval_response": "_handle_approval_response",
        "user_question_response": "_handle_user_question_response",
        "batch_review_accept": "_handle_batch_review_accept",
        "batch_review_revert": "_handle_batch_review_revert",
        "batch_review_accept_file": "_handle_batch_review_accept_file",
        "batch_review_revert_file": "_handle_batch_review_revert_file",
    }

    async def _resend_parked_prompt(self) -> None:
        """Put back every card a turn is parked on — in any conversation.

        A turn parks on a person: an approval, a clarification, a plan awaiting
        approval. The agent blocks on that answer with no timeout — deliberately, so
        nothing proceeds because the user was slow — which means a connection that drops
        while a card is up parks that conversation for ever, with nothing on screen to
        explain the silence.

        Every conversation, not just the one on screen: each agent runs its own turn, so
        several can be parked at once, and a card left out is a turn waiting for ever.
        Nothing here judges whose conversation it belongs to — the card says so itself, and
        that is what routes the answer back to the agent that asked.
        """
        for _sid, worker in self.pool.items():
            prompt = worker.human.pending_prompt()
            if not prompt:
                continue
            try:
                await self.ws.send(json.dumps(prompt))
            except Exception:
                logger.debug("_resend_parked_prompt: the client went away mid-send", exc_info=True)
                return

    def _worker_for_answer(self, msg: dict) -> _AgentWorker | None:
        """The agent an answered card belongs to, or None if it belongs to nobody.

        The most consequential routing decision here. A card says which conversation asked
        it — several can be parked at once, and one may be asking while the user reads
        another — so the answer goes to that conversation's agent. Aiming it at whichever
        agent is on screen would settle a question a different conversation asked, with the
        user's approval attached to a tool call they never saw, which is the worst failure
        this layer can produce.

        So an unknown or missing session is **dropped**. A card carrying no session comes
        from a client that does not send one; defaulting it to the conversation on screen
        would make exactly the mistake the attribution exists to prevent.
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

    async def _handle_approval_response(self, msg: dict) -> None:
        answer = {"choice": msg.get("choice", "n"), "approved_files": msg.get("approved_files")}
        if await self._answer_deferred(msg, answer):
            return
        worker = self._worker_for_answer(msg)
        if worker is not None:
            worker.human.resolve_approval(answer["choice"], answer["approved_files"],
                                    str(msg.get("id") or ""))

    async def _handle_user_question_response(self, msg: dict) -> None:
        if await self._answer_deferred(msg, {"answers": msg.get("answers") or []}):
            return
        worker = self._worker_for_answer(msg)
        if worker is not None:
            # With the card's id: an answer reaches the card it was shown on, and one
            # naming a card that is gone settles nothing.
            worker.human.resolve_question(msg.get("answers"), str(msg.get("id") or ""))

    async def _resend_deferred_prompt(self) -> None:
        """Put the card a deferred turn of this session waits on back on screen."""
        prompt = (self.conv.pending_interaction or {}).get("prompt")
        if not prompt:
            return
        try:
            await self.ws.send(json.dumps(prompt, default=str))
        except Exception:
            logger.debug("_resend_deferred_prompt: the client went away mid-send", exc_info=True)

    async def _answer_deferred(self, msg: dict, answer: dict) -> bool:
        """Route an answer to a deferred card into a resume turn. True when handled.

        An answer to a card the user moved past is swallowed: handed to the worker, it
        would sit on the queue and settle the next prompt the user never saw.
        """
        msg_id = msg.get("id")
        if msg_id and msg_id in self._stale_prompt_ids:
            return True
        conv = self.conv
        record = conv.pending_interaction
        if not record or not msg_id or msg_id != (record.get("prompt") or {}).get("id"):
            return False
        conv.pending_interaction = None
        # Answered once: a second answer to the same card (a copy of it on screen)
        # must not settle the next live prompt either.
        self._stale_prompt_ids.add(msg_id)
        if record.get("kind") == KIND_CALLS:
            # The placeholders leave the record here as they leave the working copy
            # in the loop, so the real results land where they stood.
            take_deferred_calls(conv.history_full, record.get("call_ids") or [])
            take_deferred_calls(conv.history, record.get("call_ids") or [])
        self._save()
        conv.submitted_len = len(conv.history)
        self.worker.submit_resume(record, answer, list(conv.history),
                                  session_id=self._active_session_id)
        return True

    async def _handle_batch_review_accept(self, msg: dict) -> None:
        # User accepted all pending file edits — keep them on disk, clear snapshots.
        if self.worker._agent is not None:
            self.worker._agent.approvals.batch.snapshots.clear()
            self.worker._agent.approvals.batch.forget_reviewed()
        await self.ws.send(json.dumps({"type": "batch_status", "files": []}))

    async def _handle_batch_review_revert(self, msg: dict) -> None:
        # User wants to undo all pending file edits — restore originals.
        moved: list[str] = []
        if self.worker._agent is not None:
            approvals = self.worker._agent.approvals
            for path, original in list(approvals.batch.snapshots.items()):
                if not self._revert_one(path, original):
                    moved.append(os.path.relpath(path))
            approvals.batch.snapshots.clear()
            approvals.batch.forget_reviewed()
        await self.ws.send(json.dumps({"type": "batch_status", "files": []}))
        await self._report_unreverted(moved)

    def _revert_one(self, path: str, original: str | None) -> bool:
        """Restore *path* to *original*, unless it has moved since it was reviewed.

        A revert undoes the diff the user looked at. A file that changed since then
        carries work from outside this review — another conversation editing the same file,
        or the user's own editor — and writing the baseline over it destroys work nobody
        asked to discard. The file is left alone and reported instead.

        Returns False only for that case; a file it could not write is a best-effort
        failure, like the rest of this path.
        """
        approvals = self.worker._agent.approvals
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
            await self.ws.send(json.dumps({
                "type": "status",
                "text": ("  ⚠ Left unchanged — changed since you were shown the diff, so "
                         "reverting would discard that too: " + ", ".join(sorted(moved))),
            }))
        except Exception:
            logger.debug("_report_unreverted: the client went away mid-send", exc_info=True)

    async def _handle_batch_review_accept_file(self, msg: dict) -> None:
        # Accept a single file — remove its snapshot, keep file on disk.
        file_rel = msg.get("file", "")
        if file_rel and self.worker._agent is not None:
            target = os.path.normpath(os.path.abspath(file_rel))
            snapshots = self.worker._agent.approvals.batch.snapshots
            key = next(
                (k for k in snapshots if os.path.normpath(os.path.abspath(k)) == target),
                None,
            )
            if key is not None:
                del snapshots[key]
                self.worker._agent.approvals.batch.forget_reviewed(key)
        self.worker._push_batch_status()

    async def _handle_batch_review_revert_file(self, msg: dict) -> None:
        # Revert a single file — restore its original content.
        file_rel = msg.get("file", "")
        if file_rel and self.worker._agent is not None:
            target = os.path.normpath(os.path.abspath(file_rel))
            snapshots = self.worker._agent.approvals.batch.snapshots
            key = next(
                (k for k in snapshots if os.path.normpath(os.path.abspath(k)) == target),
                None,
            )
            if key is not None:
                original = snapshots.pop(key)
                reverted = self._revert_one(key, original)
                self.worker._agent.approvals.batch.forget_reviewed(key)
                if not reverted:
                    await self._report_unreverted([os.path.relpath(key)])
        self.worker._push_batch_status()

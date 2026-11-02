"""Editing a message already sent: the conversation goes back to just before it.

A handler group of ``Connection``. ``rewind_preview`` answers with what a rewind would
do; ``rewind`` does it (turn stopped, files put back, runs stopped, conversation cut,
carried context and checklist restored), then answers ``rewound`` with the cut
transcript, on which the client sends the edited message as a new query.
"""
from __future__ import annotations

import asyncio
import logging
import os

from .....context.checkpoints import RewindPlan
from .....context.todo import write_todo_items
from ...runtime import todo_file_for_session
from .base import Handlers

logger = logging.getLogger(__name__)

# How long a rewind waits for the turn it cancelled to end, in steps of _TURN_POLL_SECS.
_TURN_STOP_SECS = 5.0
_TURN_POLL_SECS = 0.02
# How long the rewind waits on the runs' cancel ops, all at once. Their watches are
# dropped at the start, so one that answers later finishes on its own.
_RUN_STOP_SECS = 2.0


class RewindHandlers(Handlers):
    MESSAGES: dict[str, str] = {
        "rewind_preview": "on_rewind_preview",
        "rewind": "on_rewind",
    }

    def _plan(self, msg: dict) -> RewindPlan | None:
        if msg.get("session_id") and msg.get("session_id") != self.view.active_session_id:
            return None
        checkpoints = self.screen.conv.checkpoints
        message_id = str(msg.get("message_id") or "")
        if checkpoints is None or not message_id:
            return None
        return checkpoints.plan(message_id)

    async def _refuse(self, message_id: str, reason: str = "") -> None:
        await self.out.send({
            "type": "rewind_plan", "message_id": message_id, "available": False,
            "session_id": self.view.active_session_id, "reason": reason,
        })

    async def on_rewind_preview(self, msg: dict) -> None:
        plan = self._plan(msg)
        if plan is None:
            await self._refuse(str(msg.get("message_id") or ""))
            return
        live = set(self._live_run_keys())
        preview = plan.preview()
        preview["runs"] = [r for r in preview["runs"] if r["job_key"] in live]
        await self.out.send({
            "type": "rewind_plan", "available": True,
            "session_id": self.view.active_session_id,
            "turn_running": self.screen.running_turn_is_ours(),
            **_relative(preview),
        })

    def _live_run_keys(self) -> list[str]:
        worker = self.pool.get(self.view.active_session_id)
        return worker.watched_job_keys() if worker is not None else []

    async def on_rewind(self, msg: dict) -> None:
        message_id = str(msg.get("message_id") or "")
        if self._plan(msg) is None:
            await self._refuse(message_id)
            return
        conv = self.screen.conv
        session_id = conv.id
        # Nothing owed to the turns about to go may start another one.
        conv.pending_wakes = []
        conv.held_checkin = None
        if not await self._stop_turn(session_id):
            # Its answer would land after the cut: nothing is touched.
            await self._refuse(message_id, "The turn in progress did not stop. Nothing "
                                           "was changed; try again once it has ended.")
            return
        # Worked out again: the turn that just stopped may have written since the preview.
        plan = self._plan(msg)
        if plan is None:
            await self._refuse(message_id)
            return
        stopped, unstopped, stopping = await self._stop_runs(plan)
        restored, failed = self._restore_files(plan)

        checkpoint = plan.checkpoint
        conv.rewind(checkpoint)
        worker = self.pool.get(session_id)
        if worker is not None:
            worker.load_agent_state({"carry_context": checkpoint.get("carry_context") or {}})
        try:
            write_todo_items(todo_file_for_session(session_id),
                             list(checkpoint.get("todos") or []),
                             list(checkpoint.get("todo_deps") or []))
        except Exception:
            logger.warning("rewind: the checklist of %s could not be put back", session_id,
                           exc_info=True)
        checkpoints = conv.checkpoints
        if checkpoints is not None:
            checkpoints.drop_from(message_id)

        summary = _summary(restored, failed, plan, stopped, unstopped, stopping)
        if summary:
            conv.note(summary)
        # What the client already rendered of the turns that are gone must not come back
        # in a replay: the watermark moves past them.
        self.pool.bus.record_client_event(session_id, {"type": "rewind",
                                                       "message_id": message_id})
        seq = self.pool.bus.last_seq(session_id)
        self.view.rendered_seq = max(self.view.rendered_seq, seq)
        if self.view.sub is not None:
            self.view.sub.rendered_through(session_id, seq)
        self.view.transcript_sync.pop(session_id, None)
        self.screen.save()

        # The cut transcript travels with ``rewound``: a ``session_loaded`` of the same
        # conversation keeps whichever copy holds more, which is the one being cut.
        await self.out.send({
            "type": "rewound", "session_id": session_id, "message_id": message_id,
            "text": str(msg.get("text") or ""),
            "display_messages": conv.display,
            "todos": conv.record.todos,
            "restored": [_rel(p) for p in restored],
            "failed": [_rel(p) for p in failed],
            "conflicts": [_rel(p) for p in plan.conflicts],
            "unrestorable": [_rel(p) for p in plan.unrestorable],
            "unstopped": unstopped,
            "stopping": stopping,
        })
        if worker is not None:
            worker.review.push_batch_status()
        self.view.last_context_usage = None
        await self.screen.emit_context_usage()

    async def _stop_turn(self, session_id: str) -> bool:
        """Cancel the turn in flight and wait for it; False when it did not end in time.

        Its answer is committed now, before the conversation is cut, and what it queued
        for this client is dropped: the cut transcript is sent whole instead.
        """
        worker = self.pool.get(session_id)
        if worker is None:
            return True
        if worker.is_busy():
            worker.cancel()
        waited = 0.0
        while worker.is_busy() and waited < _TURN_STOP_SECS:
            await asyncio.sleep(_TURN_POLL_SECS)
            waited += _TURN_POLL_SECS
        if worker.is_busy():
            logger.warning("rewind: the turn of %s did not stop in %.0fs", session_id,
                           _TURN_STOP_SECS)
            return False
        self.pool.bus.pump_once()
        sub = self.view.sub
        if sub is None:
            return True
        kept = []
        while not sub.queue.empty():
            ev, extras = sub.queue.get_nowait()
            if ev.get("session_id") != session_id:
                kept.append((ev, extras))
        for item in kept:
            sub.queue.put_nowait(item)
        return True

    async def _stop_runs(self, plan: RewindPlan) -> tuple[list[str], list[str], list[str]]:
        """Stop the runs started since the message, all at once; (stopped, refused, still stopping)."""
        worker = self.pool.get(self.view.active_session_id)
        if worker is None:
            return [], [], []
        live = set(worker.watched_job_keys())
        keys = [str(r.get("job_key") or "") for r in plan.runs]
        pending = {asyncio.wrap_future(worker.withdraw_run(k)): k for k in keys if k in live}
        if not pending:
            return [], [], []
        done, late = await asyncio.wait(pending, timeout=_RUN_STOP_SECS)
        unstopped: list[str] = []
        for future in late:
            future.add_done_callback(_log_late_stop(pending[future]))
        for future in done:
            try:
                payload = future.result()
            except Exception:
                logger.warning("rewind: run %s could not be stopped", pending[future],
                               exc_info=True)
                unstopped.append(pending[future])
                continue
            if isinstance(payload, dict) and payload.get("status") == "error":
                unstopped.append(pending[future])
        late_keys = [pending[f] for f in late]
        stopped = [k for k in pending.values() if k not in unstopped and k not in late_keys]
        return stopped, unstopped, late_keys

    def _restore_files(self, plan: RewindPlan) -> tuple[list[str], list[str]]:
        """Put every file of *plan* back; (restored, failed). The batch review forgets them."""
        checkpoints = self.screen.conv.checkpoints
        restored: list[str] = []
        failed: list[str] = []
        if checkpoints is None:
            return restored, failed
        for path, digest in plan.restore:
            (restored if checkpoints.restore(path, digest) else failed).append(path)
        worker = self.pool.get(self.view.active_session_id)
        agent = worker.agent if worker is not None else None
        if agent is not None:
            batch = agent.approvals.batch
            gone = {os.path.abspath(p) for p in restored}
            for key in list(batch.snapshots):
                if os.path.abspath(key) in gone:
                    del batch.snapshots[key]
                    batch.forget_reviewed(key)
        return restored, failed


def _summary(restored: list[str], failed: list[str], plan: RewindPlan,
             stopped: list[str], unstopped: list[str], stopping: list[str]) -> str:
    """The line the transcript keeps of what the rewind did, or "" when it touched nothing."""
    lines = []
    if restored:
        lines.append(f"↩ Put back {len(restored)} file(s): " + ", ".join(_rel(p) for p in restored))
    if stopped:
        lines.append("↩ Stopped: " + ", ".join(stopped))
    if stopping:
        lines.append("↩ Still stopping: " + ", ".join(stopping))
    left = [*plan.conflicts, *failed]
    if left:
        lines.append("⚠ Left as they are (changed since, or could not be written): "
                     + ", ".join(_rel(p) for p in left))
    if plan.unrestorable:
        lines.append("⚠ No copy kept, not put back: "
                     + ", ".join(_rel(p) for p in plan.unrestorable))
    if unstopped:
        lines.append("⚠ Could not be stopped: " + ", ".join(unstopped))
    return "\n".join(lines)


def _log_late_stop(job_key: str):
    def _done(future: asyncio.Future) -> None:
        if future.cancelled() or future.exception() is not None:
            logger.warning("rewind: run %s could not be stopped", job_key)
    return _done


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


def _relative(preview: dict) -> dict:
    """*preview* with its paths as the user reads them: relative to the workspace."""
    out = dict(preview)
    out["files"] = [{**f, "path": _rel(f["path"])} for f in preview["files"]]
    out["conflicts"] = [_rel(p) for p in preview["conflicts"]]
    out["unrestorable"] = [_rel(p) for p in preview["unrestorable"]]
    return out

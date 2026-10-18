"""The conversations of this connection: listing, opening, creating, deleting, saving.

A handler group of ``Connection``, built with its view state, its socket, the pool
and the shared ``Screen``. ``MESSAGES`` names the inbound message types it answers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime

from ...runtime import todo_file_for_session, write_active_session
from .....session.jobs_on_disk import scan_session
from .....session.summary import PROVISIONAL_VERSION, SUMMARY_VERSION, generate_summary
from ...pool.pool import AgentPool
from ..screen import Screen
from ..view import Sender, ViewState
from .base import Handlers

logger = logging.getLogger(__name__)


class ConversationHandlers(Handlers):
    MESSAGES: dict[str, str] = {
        "clear_todos": "on_clear_todos",
        "create_session": "on_create_session",
        "switch_session": "on_switch_session",
        "delete_session": "on_delete_session",
        "rename_session": "on_rename_session",
    }

    def __init__(self, view: ViewState, out: Sender, pool: AgentPool, screen: Screen) -> None:
        super().__init__(view, out, pool, screen)
        self.store = pool.conversations.store
        # The session-summary refresh in flight, at most one per connection.
        self.summary_task: asyncio.Task[None] | None = None

    def purge_empty_sessions(self) -> None:
        """Delete sessions that have no title and no messages (stale from old reconnects)."""
        try:
            for meta in self.store.list_sessions():
                if meta.title:
                    continue
                try:
                    full = self.store.load_session(meta.id)
                    if not full.display_messages and not full.llm_history:
                        self.pool.conversations.delete(meta.id)
                except Exception:
                    logger.debug("_purge_empty_sessions: a best-effort step failed and was skipped", exc_info=True)
        except Exception:
            logger.debug("_purge_empty_sessions: a best-effort step failed and was skipped", exc_info=True)

    async def create_new_session(self) -> None:
        # Not saved yet — only once it has something in it, or every reconnect would
        # leave an empty "New session" behind.
        self.view.conv = session = self.pool.conversations.new()
        self.view.active_session_id = session.id
        self.pool.set_active(session.id)
        write_active_session(session.id)
        # A new conversation has rendered nothing. Carried over, the previous one's
        # watermark is a number from a journal this session does not have: its own
        # starts at 1, every stamped event is at or below the inherited mark, and the
        # replay gate drops the lot — while `token` and `thinking`, which are never
        # journaled and so never stamped, keep arriving. That is a chat that streams
        # text and reasoning for ever and shows no tool call, no diff and no answer.
        self.view.rendered_seq = 0
        # No agent state to load and no grants to drop: a new conversation has no agent
        # yet, and the one built for it on its first query starts empty by construction.
        # Clearing anything here would reach into another conversation's agent, and
        # truncating an approved-paths file would revoke grants its turn is writing under.
        try:
            await self.out.send({
                "type": "session_loaded",
                "session_id": session.id,
                "title": session.title,
                "display_messages": [],
                "todos": [],
                "agent_ready": self.pool.ui_state(session.id)["agent_ready"],
            })
        except Exception:
            logger.debug("_create_new_session: the client went away mid-send", exc_info=True)
        self.view.last_context_usage = None  # client cleared its bar on session_loaded
        await self.screen.emit_context_usage()

    async def load_session(self, session_id: str) -> None:
        conv = self.pool.conversations.get(session_id)
        if conv is None:
            raise FileNotFoundError(f"no conversation {session_id}")
        session = conv.record
        # Whether this conversation already has an agent decides what gets restored
        # below: an agent that has been working holds the live carry context, and reloading
        # the saved snapshot over it throws away everything it learned since. Read before
        # the pointer moves, while `worker` still resolves the one being left.
        resuming_live_agent = self.pool.get(session.id) is not None
        self.view.conv = conv
        self.view.active_session_id = session.id
        self.pool.set_active(session.id)
        # Nothing is reset on a switch. Each conversation's approved paths are its own
        # file, so there is nothing of another conversation's to drop — and dropping
        # this one's would revoke grants its own turn may still be writing under.
        write_active_session(session.id)
        self.view.rendered_seq = session.rendered_seq
        self.screen.publish_title(session.id, session.title)
        if not resuming_live_agent:
            # Only when this conversation has no agent yet — the snapshot is how a
            # rebuilt agent picks up where the last one left off. An agent that is
            # already live has moved past it.
            self.screen.worker.load_agent_state({"carry_context": session.carry_context})

        # If session had todos, restore them to disk so agent picks them up —
        # unless what is already on disk is newer than this snapshot (see below).
        #
        # And then simply show them, without asking whether to resume: this runs on
        # every session load, a reconnect mid-turn included, and the plan of record is
        # on disk either way (the agent reads the file, not this message).
        if session.todos:
            if not self._restore_todos(
                session.todos, getattr(session, 'todo_deps', None),
                snapshot_at=session.updated_at,
            ):
                try:
                    await self.out.send({
                        "type": "status",
                        "text": ("  ⓘ Kept the checklist already on disk — it is newer "
                                 "than this session's saved copy."),
                    })
                except Exception:
                    logger.debug("_load_session: the client went away mid-send", exc_info=True)
        try:
            await self.out.send({
                "type": "session_loaded",
                "session_id": session.id,
                "title": session.title,
                "display_messages": conv.display,
                "todos": session.todos,
                # A turn of this conversation that outlived the last connection is
                # still producing. Said here rather than left to be inferred: the
                # client shows the run as live, and — the part that is not
                # cosmetic — knows a turn is open, so the answer that ends it is
                # what hands the finished transcript back for saving.
                "turn_running": self.screen.running_turn_is_ours(),
                "agent_ready": self.pool.ui_state(session.id)["agent_ready"],
            })
        except Exception:
            logger.debug("_load_session: the client went away mid-send", exc_info=True)
        await self.screen.resend_deferred_prompt()
        await self._remind_untracked_jobs(session.id)
        self.view.last_context_usage = None  # client cleared its bar on session_loaded
        await self.screen.emit_context_usage()

    async def _remind_untracked_jobs(self, session_id: str) -> None:
        """Say when this conversation has runs going that nothing watches.

        A line, not a re-registration: asking about the run is what re-arms it.
        """
        worker = self.pool.get(session_id)
        if worker is not None and worker.watched_job_keys():
            return  # a watcher is holding them; the wake is coming
        live = self._live_jobs_of(session_id)
        if not live:
            return
        what = "run is" if len(live) == 1 else "runs are"
        await self.out.notify(f"⏳ {len(live)} {what} still going in this conversation, "
                           f"untracked since the agent last restarted — ask for their "
                           f"state to pick the tracking back up: {'; '.join(live[:3])}")

    def _restore_todos(
        self, todos: list[dict], todo_deps: list | None = None,
        *, snapshot_at: str = "",
    ) -> bool:
        """Write the saved checklist (and its deps) back to the todo file; True when written.

        Refused when the file is newer than *snapshot_at*: the model may have written it
        since the last save.
        """
        try:
            todo_file = todo_file_for_session(self.view.active_session_id)
            if not self._snapshot_is_current(todo_file, snapshot_at):
                return False
            os.makedirs(os.path.dirname(todo_file), exist_ok=True)
            lines = []
            for item in todos:
                check = "[x]" if item.get("done") else "[ ]"
                lines.append(f"- {check} {item.get('text', '')}")
            with open(todo_file, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            # Restore deps sidecar.
            deps_file = os.path.join(os.path.dirname(todo_file), "todo_deps.json")
            if todo_deps:
                with open(deps_file, "w", encoding="utf-8") as f:
                    json.dump(todo_deps, f)
            else:
                try:
                    os.remove(deps_file)
                except OSError:
                    pass
            return True
        except Exception:
            logger.debug("_restore_todos: a best-effort step failed and was skipped", exc_info=True)
            return False

    @staticmethod
    def _snapshot_is_current(todo_file: str, snapshot_at: str) -> bool:
        """Whether a snapshot taken at *snapshot_at* may overwrite *todo_file* (yes when unreadable)."""
        if not snapshot_at:
            return True
        try:
            snapshot_ts = datetime.fromisoformat(snapshot_at).timestamp()
        except Exception:
            logger.debug("_snapshot_is_current: a best-effort step failed and was skipped", exc_info=True)
            return True
        try:
            return os.path.getmtime(todo_file) <= snapshot_ts
        except OSError:
            return True  # nothing on disk to lose

    def schedule_summary_refresh(self) -> None:
        """Regenerate the session's description in the background, at most one at a time."""
        if self.view.active_session_id is None:
            return
        if self.summary_task is not None and not self.summary_task.done():
            return
        try:
            self.summary_task = asyncio.get_event_loop().create_task(
                self._refresh_session_summary(self.view.active_session_id)
            )
        except Exception:
            logger.debug("_schedule_summary_refresh: a best-effort step failed and was skipped", exc_info=True)

    async def _refresh_session_summary(self, session_id: str) -> None:
        try:
            conv = self.pool.conversations.get(session_id)
            if conv is None or not conv.saved:
                return
            session = conv.record
            messages = list(conv.display)
            if not messages:
                return
            # The description is built from prose only, so its freshness has to be
            # measured in prose too: counting every message would re-summarize on each
            # block of tool rows, which changes nothing the summary can see.
            text_msgs = self.screen.text_count(messages)
            # Refresh unless the stored description already covers exactly this
            # transcript and came from the current generator. A provisional one
            # (the query fallback) never counts as fresh, so it keeps retrying.
            fresh_enough = (
                session.summary
                and session.summary_version == SUMMARY_VERSION
                and session.summary_msgs >= text_msgs
            )
            if fresh_enough:
                return
            summary, generated = await asyncio.get_event_loop().run_in_executor(
                None, generate_summary, self.screen.worker.model, messages
            )
            if not summary:
                return
            # The same object the turns write, so nothing they wrote meanwhile is lost.
            session.summary = summary
            session.summary_msgs = text_msgs
            session.summary_version = SUMMARY_VERSION if generated else PROVISIONAL_VERSION
            self.pool.conversations.save(conv)
            await self.screen.send_sessions_list()
        except Exception:
            logger.debug("_refresh_session_summary: a best-effort step failed and was skipped", exc_info=True)
            pass  # Descriptions are cosmetic — never disturb the session.

    async def on_clear_todos(self, msg: dict) -> None:
        self.screen.worker.review.clear_todos()
        await self.out.send({"type": "todo", "items": []})

    async def on_create_session(self, msg: dict) -> None:
        # The turn running in the conversation left keeps running: its answer goes to
        # that conversation, which is the pool's to write, not this view's.
        self.screen.save()
        await self.create_new_session()
        await self.screen.send_sessions_list()

    async def on_switch_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        if not target_id or self.pool.conversations.get(target_id) is None:
            return
        if target_id == self.view.active_session_id:
            return
        self.screen.save()
        await self.load_session(target_id)
        await self.screen.send_sessions_list()

    def _live_work_of(self, session_id: str) -> list[str]:
        """Everything deleting *session_id* would cut short: its turn, its card, its runs."""
        live: list[str] = []
        if self.pool.is_parked(session_id):
            live.append("a turn waiting for your answer")
        elif self.pool.is_busy(session_id):
            live.append("a turn in progress")
        return live + self._live_jobs_of(session_id)

    def _live_jobs_of(self, session_id: str) -> list[str]:
        """Commands of *session_id* still running, by the scan's rule (``jobs_on_disk``)."""
        return [(job.command or job.job_key)[:80]
                for job in scan_session(session_id, include_reported=True) if job.live]

    async def on_delete_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        if not target_id:
            return
        running = self._live_work_of(target_id)
        if running and not msg.get("force") and target_id not in self.view.delete_refused:
            self.view.delete_refused.add(target_id)
            # Said rather than done, on the rendered channel (status text is dropped as chatter).
            await self.screen.command_reply(
                "delete",
                "Not deleted — that conversation still has work running",
                items=[{"label": label} for label in running[:5]],
                note=(f"(+{len(running) - 5} more) " if len(running) > 5 else "")
                     + "Stop it first, or delete again to discard it.",
                tone="warn",
            )
            await self.screen.send_sessions_list()
            return
        self.view.delete_refused.discard(target_id)
        was_active = (target_id == self.view.active_session_id)
        self.view.forget_session(target_id)
        # Its agent goes with it, freeing a slot — and whatever it was queued to run,
        # which would otherwise be admitted into a conversation that is gone.
        await self.pool.close(target_id)
        # The store removes its sidecar directory too (todo_list.md, plan.md, …).
        self.pool.conversations.delete(target_id)
        if was_active:
            await self.create_new_session()
        await self.pool.pump()   # a freed slot may admit a conversation that was waiting
        await self.screen.send_sessions_list()

    async def on_rename_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        new_title = (msg.get("title") or "").strip()
        conv = self.pool.conversations.get(target_id)
        if conv is None:
            return
        conv.record.title = new_title
        conv.record.title_custom = True  # a hand-picked title outranks the generated description
        self.pool.writer.save_conversation(conv)
        self.screen.publish_title(target_id, new_title)
        await self.screen.send_sessions_list()

"""The conversations of this connection: listing, opening, creating, deleting, saving.

A part of ``Connection``: a mixin over the connection's own state, so a
handler reads and writes the session it serves. ``MESSAGES`` names the inbound message
types it answers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime

from ...runtime import _todo_file_for_session, _write_active_session
from .....session.jobs_on_disk import is_settled
from .....session.summary import PROVISIONAL_VERSION, SUMMARY_VERSION, generate_summary
from ......servers._shared.state_paths import session_dir
from .....config import paths

logger = logging.getLogger(__name__)


class ConversationHandlers:
    MESSAGES: dict[str, str] = {
        "clear_todos": "_handle_clear_todos",
        "create_session": "_handle_create_session",
        "switch_session": "_handle_switch_session",
        "delete_session": "_handle_delete_session",
        "rename_session": "_handle_rename_session",
    }

    def _purge_empty_sessions(self) -> None:
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

    async def _send_sessions_list(self) -> None:
        sessions = self.store.list_sessions()
        # The conversation on screen before it is saved, so the panel shows it at once.
        if self.conv is not None and not self.conv.saved:
            if self.conv.id not in {s.id for s in sessions}:
                sessions = [self.conv.meta()] + sessions
        try:
            await self.ws.send(json.dumps({
                "type": "sessions_list",
                "sessions": [{**s.to_dict(), **self._session_activity(s.id)}
                             for s in sessions],
            }))
        except Exception:
            logger.debug("_send_sessions_list: the client went away mid-send", exc_info=True)

    def _session_activity(self, session_id: str) -> dict:
        """What a conversation is doing right now, for its row in the panel.

        The part of concurrency the user cannot do without. A turn running in a
        conversation nobody is reading is invisible otherwise, and one *parked on a card* is
        worse than invisible: it waits on a person, with no timeout, by design — so a
        conversation can sit stopped for ever with nothing on screen to explain it.

        ``runs`` is the count of background runs it still has going, and it is not the
        same question as ``running``: a conversation that launched a two-hour build and
        answered has no turn in flight at all. That is the case detaching exists for,
        and it was the one case nothing reported — the panel asked whether to keep the
        work going only when a *turn* was live, so a job submitted and left behind was
        abandoned with no question asked. Read off the live worker, which is where the
        watchers are: a run cannot be live without one, since ``releasable`` refuses to
        close a worker that holds a watcher.
        """
        # Guarded: this is read for every row of every listing, and the send around it
        # swallows exceptions — so a hiccup here would blank the whole panel rather than
        # lose one dot.
        try:
            worker = self.pool.get(session_id)
            runs = len(worker.watched_job_keys()) if worker is not None else 0
            return {
                "running": self.pool.is_busy(session_id),
                "parked": self.pool.is_parked(session_id),
                "queued": self.pool.queued_position(session_id) is not None,
                "runs": runs,
            }
        except Exception:
            logger.debug("_session_activity: a best-effort step failed and was skipped", exc_info=True)
            return {}

    async def _create_new_session(self) -> None:
        # Not saved yet — only once it has something in it, or every reconnect would
        # leave an empty "New session" behind.
        self.conv = session = self.pool.conversations.new()
        self._active_session_id = session.id
        self.pool.set_active(session.id)
        _write_active_session(session.id)
        # A new conversation has rendered nothing. Carried over, the previous one's
        # watermark is a number from a journal this session does not have: its own
        # starts at 1, every stamped event is at or below the inherited mark, and the
        # replay gate drops the lot — while `token` and `thinking`, which are never
        # journaled and so never stamped, keep arriving. That is a chat that streams
        # text and reasoning for ever and shows no tool call, no diff and no answer.
        self._rendered_seq = 0
        # No agent state to load and no grants to drop: a new conversation has no agent
        # yet, and the one built for it on its first query starts empty by construction.
        # Clearing anything here would reach into another conversation's agent, and
        # truncating an approved-paths file would revoke grants its turn is writing under.
        try:
            await self.ws.send(json.dumps({
                "type": "session_loaded",
                "session_id": session.id,
                "title": session.title,
                "display_messages": [],
                "todos": [],
                "agent_ready": self.pool.ui_state(session.id)["agent_ready"],
            }))
        except Exception:
            logger.debug("_create_new_session: the client went away mid-send", exc_info=True)
        self._last_context_usage = None  # client cleared its bar on session_loaded
        await self._emit_context_usage()

    async def _load_session(self, session_id: str) -> None:
        conv = self.pool.conversations.get(session_id)
        if conv is None:
            raise FileNotFoundError(f"no conversation {session_id}")
        session = conv.record
        # Whether this conversation already has an agent decides what gets restored
        # below: an agent that has been working holds the live carry context, and reloading
        # the saved snapshot over it throws away everything it learned since. Read before
        # the pointer moves, while `worker` still resolves the one being left.
        resuming_live_agent = self.pool.get(session.id) is not None
        self.conv = conv
        self._active_session_id = session.id
        self.pool.set_active(session.id)
        # Nothing is reset on a switch. Each conversation's approved paths are its own
        # file, so there is nothing of another conversation's to drop — and dropping
        # this one's would revoke grants its own turn may still be writing under.
        _write_active_session(session.id)
        self._rendered_seq = session.rendered_seq
        self._publish_title(session.id, session.title)
        if not resuming_live_agent:
            # Only when this conversation has no agent yet — the snapshot is how a
            # rebuilt agent picks up where the last one left off. An agent that is
            # already live has moved past it.
            self.worker.load_agent_state({"carry_context": session.carry_context})

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
                    await self.ws.send(json.dumps({
                        "type": "status",
                        "text": ("  ⓘ Kept the checklist already on disk — it is newer "
                                 "than this session's saved copy."),
                    }))
                except Exception:
                    logger.debug("_load_session: the client went away mid-send", exc_info=True)
        try:
            await self.ws.send(json.dumps({
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
                "turn_running": self._running_turn_is_ours(),
                "agent_ready": self.pool.ui_state(session.id)["agent_ready"],
            }))
        except Exception:
            logger.debug("_load_session: the client went away mid-send", exc_info=True)
        await self._resend_deferred_prompt()
        await self._remind_untracked_jobs(session.id)
        self._last_context_usage = None  # client cleared its bar on session_loaded
        await self._emit_context_usage()

    async def _remind_untracked_jobs(self, session_id: str) -> None:
        """Say when this conversation has runs going that nothing is watching.

        A watcher lives on the agent that launched the run, so it goes when that agent
        goes — which the VS Code extension causes on every connect, tearing the server
        down and respawning it. The run itself does not care: it is a detached process
        with its own session directory, and it will finish whether or not anything is
        listening. What is lost is only the promise to say so.

        Deliberately a line and not a re-registration. Re-arming every run found on disk
        at load would poll jobs the user has long stopped caring about, in every
        conversation they open; asking for their state is what re-arms them (the status
        ops hand back a descriptor while a run is in flight), and that is a decision
        that belongs to the person, not to a reconnect.
        """
        worker = self.pool.get(session_id)
        if worker is not None and worker.watched_job_keys():
            return  # a watcher is holding them; the wake is coming
        live = self._live_jobs_of(session_id)
        if not live:
            return
        what = "run is" if len(live) == 1 else "runs are"
        await self._notify(f"⏳ {len(live)} {what} still going in this conversation, "
                           f"untracked since the agent last restarted — ask for their "
                           f"state to pick the tracking back up: {'; '.join(live[:3])}")

    def _restore_todos(
        self, todos: list[dict], todo_deps: list | None = None,
        *, snapshot_at: str = "",
    ) -> bool:
        """Write todo items back to the session-scoped todo file. Returns whether it did.

        Also restores the todo_deps.json sidecar when deps are provided.

        REFUSES to write over a file that is newer than *snapshot_at*, the session's
        ``updated_at``. ``session.todos`` is captured at autosave time, so a run cut
        short — a dropped connection mid-query — leaves the store holding the checklist
        as it stood at the last save while the file on disk holds the one the model has
        since written. Restoring unconditionally then destroys the live list and hands
        the model a finished one: observed in session ba8eee87, where a disconnect
        during an optimisation task restored the *previous* task's five completed steps,
        so the next query ran with a plan of record reading "0 pending" and the
        completion gate (``unchecked_checklist_items``) had nothing left to block on.

        The snapshot is trusted when it is at least as recent as the file, when the file
        does not exist, and when either timestamp cannot be read — restoring is the old
        behaviour and the right default; only *destroying newer work* is refused. The
        deps sidecar follows the same decision as the list it describes, since restoring
        one without the other yields dependencies pointing at steps that are not there.
        """
        try:
            todo_file = _todo_file_for_session(self._active_session_id)
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
        """May a session snapshot taken at *snapshot_at* overwrite *todo_file*?

        Only when the file is not newer than the snapshot. ``updated_at`` is written
        with an explicit UTC offset, so its epoch is directly comparable to the file's
        mtime. Anything unreadable — no timestamp, a malformed one, a missing file —
        answers yes: the guard exists to stop one specific loss, not to become a second
        way for a restore to fail.
        """
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

    def _save(self) -> None:
        """Write the conversation on screen down, with how far this client has rendered
        it — never backwards: a replay is sized from it, and lowering it would resend
        what the client already has."""
        if self.conv is None:
            return
        record = self.conv.record
        record.rendered_seq = max(record.rendered_seq, self._rendered_seq)
        self.pool.save_conversation(self.conv)

    def _schedule_summary_refresh(self) -> None:
        """Kick off a background regeneration of the session's description.

        Called once each turn has answered, so the description covers the work and
        not just the request. Fire-and-forget: the model call runs in an executor so
        the WS event loop stays responsive, and the refreshed sessions list is pushed
        when it lands. At most one refresh is in flight per connection.
        """
        if self._active_session_id is None:
            return
        if self._summary_task is not None and not self._summary_task.done():
            return
        try:
            self._summary_task = asyncio.get_event_loop().create_task(
                self._refresh_session_summary(self._active_session_id)
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
            text_msgs = self._text_count(messages)
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
                None, generate_summary, self.worker.model, messages
            )
            if not summary:
                return
            # The same object the turns write, so nothing they wrote meanwhile is lost.
            session.summary = summary
            session.summary_msgs = text_msgs
            session.summary_version = SUMMARY_VERSION if generated else PROVISIONAL_VERSION
            self.pool.conversations.save(conv)
            await self._send_sessions_list()
        except Exception:
            logger.debug("_refresh_session_summary: a best-effort step failed and was skipped", exc_info=True)
            pass  # Descriptions are cosmetic — never disturb the session.

    def _session_title(self, session_id: str) -> str:
        """A conversation's name, or "" if it has none (or none saved yet)."""
        conv = self.pool.conversations.get(session_id)
        return conv.title if conv is not None else ""

    def _publish_title(self, session_id: str | None, title: str) -> None:
        """Tell a conversation's agent what that conversation is called.

        Only so a card it raises while the user is reading elsewhere can say who is
        asking. The agent has no other use for it, and never sees it in its own prompt.
        """
        worker = self.pool.get(session_id)
        if worker is not None:
            worker.session_title = title or ""

    async def _handle_clear_todos(self, msg: dict) -> None:
        self.worker._clear_todos()
        await self.ws.send(json.dumps({"type": "todo", "items": []}))

    async def _handle_create_session(self, msg: dict) -> None:
        # The turn running in the conversation left keeps running: its answer goes to
        # that conversation, which is the pool's to write, not this view's.
        self._save()
        await self._create_new_session()
        await self._send_sessions_list()

    async def _handle_switch_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        if not target_id or self.pool.conversations.get(target_id) is None:
            return
        if target_id == self._active_session_id:
            return
        self._save()
        await self._load_session(target_id)
        await self._send_sessions_list()

    def _live_work_of(self, session_id: str) -> list[str]:
        """Everything of *session_id*'s that deleting it would cut short.

        Its turn counts, not only its jobs: conversations run turns at once, so the one
        being deleted may be mid-task somewhere the user is not looking — and deleting it
        closes its agent under the turn, which fails its next tool call rather than ending
        it. Its card counts too, for the same reason read the other way: a turn parked on
        a person is work waiting to continue, not work that has stopped.
        """
        live: list[str] = []
        if self.pool.is_parked(session_id):
            live.append("a turn waiting for your answer")
        elif self.pool.is_busy(session_id):
            live.append("a turn in progress")
        return live + self._live_jobs_of(session_id)

    def _live_jobs_of(self, session_id: str) -> list[str]:
        """Commands of *session_id* still running, as far as the job dirs can say.

        A session's detached shell jobs and submitted Slurm jobs live in its own
        directory now, which is what gives them a retention policy — they go when the
        conversation goes. The flip side is that deleting a conversation would take the
        log of a process that is still running, and for a Slurm job still queued, the
        only record of what was submitted. So the deletion asks first.

        Read off the files rather than any in-memory registry: the job outlives the
        worker that launched it, and may outlive the server. A shell job is live when it
        wrote no ``exit_code``, a Slurm submission when no poll has recorded the state
        Slurm last gave it — ``jobs_on_disk.is_settled`` holds both halves of that question,
        because a Slurm job records no exit code and asking only for one reads every job
        a conversation ever submitted as still running. Neither is a certainty, which is
        why this reports and does not act.
        """
        base = session_dir(session_id, paths.STATE_DIR)
        live = []
        for kind in ("jobs", "hpc_jobs"):
            root = os.path.join(base, kind)
            try:
                keys = sorted(os.listdir(root))
            except OSError:
                continue
            for key in keys:
                job = os.path.join(root, key)
                # A job is a directory. Anything else here is not one, and reporting it
                # as a run with no exit code — which is to say a live one — makes
                # deleting the conversation ask about work that does not exist.
                if not os.path.isdir(job):
                    continue
                if is_settled(job):
                    continue
                label = key
                try:
                    with open(os.path.join(job, "meta.json"), encoding="utf-8") as fh:
                        label = (json.load(fh).get("command") or key)[:80]
                except (OSError, ValueError):
                    pass
                live.append(label)
        return live

    async def _handle_delete_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        if not target_id:
            return
        running = self._live_work_of(target_id)
        if running and not msg.get("force") and target_id not in self._delete_refused:
            self._delete_refused.add(target_id)
            # Said rather than done: the user may well want it gone anyway, and the
            # answer is theirs. Nothing is deleted in the meantime.
            #
            # On the rendered channel, not ``status``. The transcript drops status text
            # as transient tool chatter, so a refusal sent there is a click that does
            # nothing visible followed by one that deletes — the warning this exists to
            # give, not given.
            await self._command_reply(
                "delete",
                "Not deleted — that conversation still has work running",
                items=[{"label": label} for label in running[:5]],
                note=(f"(+{len(running) - 5} more) " if len(running) > 5 else "")
                     + "Stop it first, or delete again to discard it.",
                tone="warn",
            )
            await self._send_sessions_list()
            return
        self._delete_refused.discard(target_id)
        was_active = (target_id == self._active_session_id)
        self._forget_session_state(target_id)
        # Its agent goes with it, freeing a slot — and whatever it was queued to run,
        # which would otherwise be admitted into a conversation that is gone.
        await self.pool.close(target_id)
        # The store removes its sidecar directory too (todo_list.md, plan.md, …).
        self.pool.conversations.delete(target_id)
        if was_active:
            await self._create_new_session()
        await self.pool.pump()   # a freed slot may admit a conversation that was waiting
        await self._send_sessions_list()

    def _forget_session_state(self, session_id: str) -> None:
        """Drop everything this connection keeps per conversation, for one that is gone."""
        for per_session in (self._live_rows, self._sent_progress, self._foreign_withheld,
                            getattr(self, "_transcript_sync", None) or {}):
            per_session.pop(session_id, None)

    async def _handle_rename_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        new_title = (msg.get("title") or "").strip()
        conv = self.pool.conversations.get(target_id)
        if conv is None:
            return
        conv.record.title = new_title
        conv.record.title_custom = True  # a hand-picked title outranks the generated description
        self.pool.save_conversation(conv)
        self._publish_title(target_id, new_title)
        await self._send_sessions_list()

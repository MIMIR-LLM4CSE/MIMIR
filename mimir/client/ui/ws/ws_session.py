"""The per-connection WebSocket session for the WS server.

``_Session`` owns one WebSocket connection: which conversation it shows, the drain
loop that forwards the stream to the client, and the inbound-message dispatch table.
The conversation itself is the pool's (``session.conversation``): this view writes
what the user does to it, and renders what the agents did. Split out of
``ws_server.py``; see that module's docstring for the wire protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from . import server_registry
from ._ws_runtime import (
    _MIMIR_DIR_WS,
    _todo_file_for_session,
    _write_active_session,
    context_budget_for,
    get_backend,
)
from .detach import detach as detach_process
from .event_router import EventRouter
from .job_scan import establish_baseline, has_baseline, is_settled, scan_all_sessions
from .panel_runs import attach_runs, recorded_hosts_sections
from .transcript_log import read_since
from .ws_commands import WS_COMMANDS, WsCommandHost
from .ws_pool import _AgentPool, idle_stop_reason
from .ws_worker import _AgentWorker
from ...protocol.messages import reconcile_tool_pairs
from ...query_engine.deferral import KIND_CALLS, take_deferred_calls
from ...session.commands import CommandResult
from ...session.conversation import Conversation
from ...session.store import list_subagents as _list_subagents, read_subagent as _read_subagent
from ...tool_execution import run_channel

# The websocket connection handle passed in by the server (websockets library type
# varies by version); kept as an untyped alias so the annotation resolves.
_WS = Any

logger = logging.getLogger(__name__)


# Events that must reach the user no matter which conversation they belong to: each
# one is a question the agent is parked on, and filtering it as "foreign" (which it is,
# during a background-job wake in another session) would leave the turn waiting on an
# answer the user was never shown. They carry their ``session_id``, so the client can
# say which conversation is asking.
# ``prompt_expired`` rides with them: it takes one of these cards back off the screen,
# and filtering it as foreign would leave a card up for a wait that has ended.
_INTERACTION_EVENTS = frozenset({"approval", "user_question", "prompt_expired"})

# How many journal events one ``replay`` frame carries. The frame cap is 32 MiB, which a
# long detached run's transcript can exceed whole; chunking also lets the webview render
# the first of them while the rest are still arriving.
_REPLAY_CHUNK = 500

# The most a replay will resend. Past this the oldest are dropped and the frame says so:
# what a returning user needs is the end of the run, and an unbounded replay into the
# webview's reducer is how a twelve-hour absence wedges the panel.
_REPLAY_MAX_EVENTS = 5000


class _Session:
    """One WebSocket connection — owns a chat history and talks to the worker."""

    def __init__(self, ws: "_WS", pool: "_AgentPool") -> None:
        self.ws = ws
        self.pool = pool
        # The conversation on screen — the pool's own object for it, shared with
        # everything else that writes it. This connection only shows it.
        self.conv: Conversation | None = None
        self.store = pool.conversations.store
        self._active_session_id: str | None = None
        # In-flight session-summary refresh (one at a time per connection).
        self._summary_task: asyncio.Task | None = None
        # Last `used_tokens` pushed to the client, so the periodic mid-turn refresh
        # only sends a frame when the number actually moved.
        self._last_context_usage: int | None = None
        # This connection's view of the stream, opened in run(). The pump owns the
        # draining; this is only where our share of it arrives.
        self._sub = None
        # Events withheld because they belong to another conversation, per session.
        # The one quiet-chat cause that is otherwise invisible — see _is_foreign_event.
        self._foreign_withheld: dict[str, int] = {}
        # How far into the journal the client's rendered transcript goes, as the client
        # last said. The watermark a re-attach replays from.
        self._rendered_seq = 0
        # Tool rows of the running turn that publish a run channel, as
        # {call id: tool name}. The name IS the channel (see run_channel), so this is
        # what lets a divert click name the run the user is actually looking at, and
        # what the progress poller walks. Registry data passing through: nothing here
        # compares a name against a literal.
        # Keyed by session: with turns running concurrently, two conversations can each
        # have a divertible run in flight, and one map between them would let a divert
        # click detach the other one's command.
        self._live_rows: dict[str, dict[str, str]] = {}
        # Last (phase, percent) pushed per call id, so a tick that learned nothing
        # sends nothing.
        self._sent_progress: dict[str, dict[str, tuple]] = {}
        # Ids of deferred cards the user moved past without answering. Their card may
        # still be on screen; an answer to one must not reach the next live prompt.
        self._stale_prompt_ids: set[str] = set()
        # Sessions whose deletion was refused once because work of theirs was still
        # running. Asking again goes through: the warning is there to be read, not to
        # make the conversation undeletable, and the second click is the confirmation.
        self._delete_refused: set[str] = set()
        # The panel report in flight, and whether another ask (and a re-read) came in
        # behind it. Asks arriving while one is served are folded into one more pass.
        self._panel_task: asyncio.Task | None = None
        self._panel_again: dict | None = None

    @property
    def worker(self) -> _AgentWorker:
        """The agent of the conversation on screen, or None if it has none yet.

        A property, not a field: there is one worker per conversation, built on that
        conversation's first query, and every ``self.worker.*`` call site asks about the
        session being read — which is what this answers.

        Until that first query the conversation has no worker, and this answers with the
        pool's detached stand-in rather than None: every getter on it gives the right
        pre-agent answer and every setter is a no-op. Two things must not be silently
        dropped that way — submitting a query, and a setting that has to survive into the
        next worker built — and both go through the pool explicitly.
        """
        return self.pool.worker_or_detached(self._active_session_id)

    def _greeting(self) -> dict:
        """The ``ready`` sent the moment the socket is accepted.

        Its ``agent_ready`` is the part worth stating rather than leaving to be
        inferred. This is sent well before the agent exists — the worker is still
        waiting on the LLM backend, minutes of it on a cold vLLM — and the worker
        sends a second greeting, spelled "ready" too, once it really is up. The two
        were indistinguishable to the client, so the chat invited the user to type
        into an agent that could not answer. The worker's own says True; this one
        says what is actually so.
        """
        return {"type": "ready", **self.pool.ui_state(self._active_session_id)}

    async def run(self) -> None:
        # Subscribe BEFORE anything else, and this order is the point: from here on
        # nothing a turn produces can be missed, because it accumulates in this
        # subscription's queue while the rest of the handshake runs. Nothing has to be
        # swept either — the pump has already drained and journaled everything, so what
        # a worker holds is never debris.
        self._sub = self.pool.bus.subscribe()
        # Named out here, created inside the guarded body below: the ``finally`` has to
        # be able to cancel them whether or not the handshake reached the line that
        # starts them.
        drain_task: asyncio.Task | None = None
        jobs_task: asyncio.Task | None = None
        # Everything from the subscription on is guarded, and that is the point. A
        # subscription is what tells the pool a socket will route this process's output
        # — the committer and the headless wake consumer both stand down while one
        # exists — so a view that ends without closing it leaves the server believing
        # somebody is watching for as long as it lives: no turn committed, no finished
        # run handed to its conversation, no check-in. Every step of the handshake sends
        # on a socket that can close under it (a window shut mid-replay raises here),
        # which makes an unguarded exit the ordinary case rather than the exotic one.
        try:
            # Send ready immediately so the webview transitions out of "connecting".
            greeting = self._greeting()
            try:
                await self.ws.send(json.dumps(greeting))
            except Exception:
                return

            # ── Session initialisation ────────────────────────────────────────────
            # Purge any empty sessions left over from previous (pre-fix) reconnects.
            self._purge_empty_sessions()
            await self._send_sessions_list()
            await self._send_toggles()

            sessions = self.store.list_sessions()
            if sessions:
                # Auto-load the most recent session.
                try:
                    await self._load_session(sessions[0].id)
                except Exception:
                    await self._create_new_session()
            else:
                await self._create_new_session()

            # What happened while nobody was attached, before the cards: a replay under a
            # chat that is already on screen, and a card on top of that.
            await self._send_replay()

            # The run never stopped, so its own level is the truth the panel must show.
            await self._restore_detached_autonomy()

            # Last, so the card lands under a chat that is already on screen.
            await self._resend_parked_prompt()

            await self._send_served_models()

            # The greeting above may have said "not ready" and the worker may have come up
            # since. It announces that once, by queueing a second ``ready``, and an
            # announcement made before this subscription existed is one nothing will repeat:
            # the chat would sit on "starting the agent" for the life of the socket with a
            # working agent behind it. Asking the worker directly costs one message.
            if not greeting["agent_ready"] and self.worker.agent_ready():
                try:
                    await self.ws.send(json.dumps(self._greeting()))
                except Exception:
                    return

            drain_task = asyncio.create_task(self._drain_loop())
            # The runs that ended while nothing was listening, off the critical path and
            # deliberately so. It walks every session's job directories, and a job
            # directory is never swept however old — so on a long-lived workspace this is
            # unbounded file work. Awaited here, inside the handshake, it sat between the
            # client connecting and this loop reading its first message: the chat came up,
            # said it was ready, and the query was never read. A task cannot do that.
            jobs_task = asyncio.create_task(self._report_ended_jobs())
            try:
                async for raw in self.ws:
                    await self._handle(raw)
            except Exception:
                pass
        finally:
            # Asked for rather than read: this block runs on every way out, including
            # a greeting that never reached the socket, and cleanup that raises here
            # strands the subscription it is on its way to close.
            summary_task = getattr(self, "_summary_task", None)
            if summary_task is not None:
                summary_task.cancel()
            for task in (drain_task, jobs_task):
                if task is None:
                    continue
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            # The agents and the pump live on; only this view of them ends here.
            self._sub.close()

    # ── Session helpers ───────────────────────────────────────────────────────

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
                    pass
        except Exception:
            pass

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
            prompt = worker.pending_prompt()
            if not prompt:
                continue
            try:
                await self.ws.send(json.dumps(prompt))
            except Exception:
                return

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
            pass

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
            pass
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
        # And then simply show them. A checklist with work left on it used to raise a
        # modal asking whether to resume the plan or start fresh, which was the wrong
        # question asked at the worst moment: this runs on every session load, and a
        # connection dropping mid-turn reloads the session — so the card arrived in the
        # middle of the very work it was asking whether to continue. There is no reading
        # of "start fresh" that is right there, and the plan of record is on disk either
        # way: the agent reads the file, not this message.
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
                    pass
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
            pass
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
                import json as _json
                with open(deps_file, "w", encoding="utf-8") as f:
                    _json.dump(todo_deps, f)
            else:
                try:
                    os.remove(deps_file)
                except OSError:
                    pass
            return True
        except Exception:
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
            from datetime import datetime
            snapshot_ts = datetime.fromisoformat(snapshot_at).timestamp()
        except Exception:
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

    # ── Session summary ───────────────────────────────────────────────────────

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
            pass

    async def _refresh_session_summary(self, session_id: str) -> None:
        try:
            from .session_summary import PROVISIONAL_VERSION, SUMMARY_VERSION, generate_summary
        except ImportError:
            from mimir.client.ui.ws.session_summary import (
                PROVISIONAL_VERSION, SUMMARY_VERSION, generate_summary,
            )
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
            pass  # Descriptions are cosmetic — never disturb the session.

    # ── Context-budget helpers ────────────────────────────────────────────────

    def _ctx_budget(self) -> tuple:
        """Return (total_tokens, reserved_tokens) for the current context mode.

        For vLLM the window tracks the server's reported max_model_len (primed at
        startup so this runs against the cache, never blocking the event loop).
        """
        mode = self.worker.setting("context_mode", self.conv.record.context_mode)
        total, reserved, _, _ = context_budget_for(self.worker.model, mode)
        return total, reserved

    async def _emit_context_usage(self) -> None:
        """Push a context_usage event to the WS client (best-effort, never raises)."""
        try:
            total, reserved = self._ctx_budget()
            # While a query runs, the transcript that matters is the agent's in-flight
            # one: the conversation only gains the turn once the answer lands, so the bar
            # would sit frozen for the whole run.
            messages = self.worker.live_history()
            if messages is None:
                messages = self.conv.history
            # allow_network=False: on the WS event loop, which must not block on a
            # tokenize round-trip. Already-counted messages hit the shared cache for
            # exact numbers; the rest fall back to the heuristic.
            history_used = get_backend().count_messages_tokens(
                self.worker.model, messages, allow_network=False
            )
            # Include the fixed per-call overhead (system prompt + tools schema).
            # Without it the bar shows only the conversation and hides the tens of
            # thousands of tokens (~30k with every tool server on) that ride on each call.
            overhead = self.worker.context_overhead_tokens()
            used = history_used + overhead
            if used == self._last_context_usage:
                return  # nothing moved — don't spend a frame on an identical payload
            self._last_context_usage = used
            await self.ws.send(json.dumps({
                "type": "context_usage",
                "used_tokens": used,
                "total_tokens": total,
                "reserved_tokens": reserved,
                "overhead_tokens": overhead,
                "overhead_measured": self.worker.context_overhead_is_measured(),
                # No agent yet, so the fixed part — tens of thousands of tokens — is not
                # in `used_tokens` at all. Said out loud rather than left to be read off
                # a missing figure: this is the one moment the bar is too low by more
                # than a rounding error, and a bar that claims an overflow it cannot
                # have measured teaches the user to distrust the one that can.
                "provisional": overhead == 0,
                # What the model actually has this turn, against the untrimmed record
                # kept behind it — so a trimmed window is visible, not silent.
                "history_messages": len(self.conv.history),
                "history_messages_full": len(self.conv.history_full),
            }))
        except Exception:
            pass

    # ── Drain loop ────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        """Forward this connection's share of the stream to the client (``EventRouter``).

        ``run()`` subscribes before the handshake so nothing can be missed; the router
        subscribes itself for a caller that starts the loop on its own.
        """
        await EventRouter(self).run()

    # Maps an inbound WS message "type" to the _Session handler method that serves it.
    # Replaces the former 13-branch ``if mtype == …`` chain in _handle.
    _MSG_HANDLERS: dict[str, str] = {
        "query": "_handle_query",
        "transcript": "_handle_transcript",
        "steer": "_handle_steer",
        "detach": "_handle_detach",
        "shutdown": "_handle_shutdown",
        "divert_to_background": "_handle_divert_to_background",
        "approval_response": "_handle_approval_response",
        "user_question_response": "_handle_user_question_response",
        "batch_review_accept": "_handle_batch_review_accept",
        "batch_review_revert": "_handle_batch_review_revert",
        "batch_review_accept_file": "_handle_batch_review_accept_file",
        "batch_review_revert_file": "_handle_batch_review_revert_file",
        "clear_todos": "_handle_clear_todos",
        "create_session": "_handle_create_session",
        "switch_session": "_handle_switch_session",
        "delete_session": "_handle_delete_session",
        "rename_session": "_handle_rename_session",
        "command": "_handle_command_msg",
        "list_toggles": "_handle_list_toggles",
        "list_resources": "_handle_list_resources",
        "toggle_server": "_handle_toggle_server",
        "toggle_skill": "_handle_toggle_skill",
        "toggle_nudge": "_handle_toggle_nudge",
        "set_model": "_handle_set_model",
        "list_subagents": "_handle_list_subagents",
        "list_panel": "_handle_list_panel",
        "read_subagent": "_handle_read_subagent",
    }

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        handler_name = self._MSG_HANDLERS.get(msg.get("type"))
        if handler_name is None:
            return  # unknown/missing type — ignore, as the old fall-through did
        try:
            await getattr(self, handler_name)(msg)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # One message failing is that message's problem. Raised out of here it
            # ended the read loop, and the connection went deaf with nothing said.
            logger.exception("ws: handling %r failed", msg.get("type"))
            try:
                await self.ws.send(json.dumps({
                    "type": "error",
                    "text": f"  ✗ {msg.get('type')} failed: {type(exc).__name__}: {exc}\n"}))
            except Exception:
                pass

    async def _announce_steer_turns(self, texts: list[str]) -> None:
        """Tell the client its unread steering is being answered by a turn of its own:
        ``steer_injected`` per message, the last one saying a turn started."""
        for i, one in enumerate(texts):
            try:
                await self.ws.send(json.dumps({
                    "type": "steer_injected", "text": one,
                    "starts_turn": i == len(texts) - 1,
                }))
            except Exception:
                return

    async def _announce_detached_answer(self, extras: dict) -> None:
        """A conversation not on screen finished a turn: say so, and that its row moved.

        The pool has already written it into its conversation, and carried on from it.
        """
        committed = extras.get("_committed") or {}
        if not committed:
            return
        title = committed.get("title") or ""
        await self._notify(f"“{title}” is waiting for your decision."
                           if committed.get("deferred")
                           else f"“{title}” finished its background turn.")
        await self._send_sessions_list()

    async def _report_ended_jobs(self) -> None:
        """Announce every detached run that finished while nothing was watching it.

        This is what makes coming back enough. A background run survives anything — its
        own process session, a trap that records the exit code, a descriptor on disk —
        but the *watcher* that promised to report it is a task on a worker's loop, and a
        restart takes it with them. Re-making that promise already existed, through a
        status tool returning ``background_jobs``, but it needed a turn in which somebody
        asked; here nobody has to.

        Handed to the pool's wake router, deliberately: a run reported late must be
        indistinguishable from one reported on time, so it goes through the same routing,
        the same wake text and the same coalescing. A conversation with no agent yet keeps
        its wake pending rather than losing it, and its next turn carries it.

        Run as a task rather than awaited in the handshake: it walks every session's
        job directories, and nothing must be able to come between a client connecting
        and the loop that reads its messages.

        Marked on disk as it is announced, because two places look for these — a
        connection arriving and a worker being built — and a run announced twice is a
        conversation woken twice for one build.

        A session never scanned before gets a *baseline* instead of a backlog. A
        detached job's directory is never swept, however old, so a long-lived workspace
        holds every build it ever ran, and an unmarked one there is as likely to have
        been reported by its own watcher as to be owed a wake. Reading that history as
        wakes owed wakes a conversation for a two-month-old build — worse than missing a
        recent one, and unlike a missed wake unbounded. Per session, and a sub-agent's
        counts as its own: a child seen here for the first time brings no backlog
        either.
        """
        try:
            found = scan_all_sessions()
        except Exception:
            logger.warning("job scan: the sessions could not be read", exc_info=True)
            return
        for session_id, jobs in found.items():
            if not has_baseline(session_id):
                establish_baseline(session_id, jobs)
                continue
            for job in jobs:
                if job.live:
                    continue   # a worker being built is what puts a watcher back on it
                descriptor = job.descriptor()
                self.pool.wakes.consume({
                    "type": "job_complete",
                    "job_key": job.job_key,
                    "server": descriptor.get("server"),
                    "kind": descriptor.get("kind"),
                    "state": job.state,
                    # Who gets woken, which for a sub-agent's run is not the session
                    # holding the files: a sub-session is a directory and no front end
                    # can load one, so a child's run is announced in the conversation
                    # that delegated it. Its markers stay under the child, where the
                    # scan that found it will look again.
                    "session_id": job.owner_session,
                    "run_session": job.session_id,
                    "summary": {"command": job.command, "exit_code": job.exit_code},
                    "status_op": job.status_op(),
                })

    async def _restore_detached_autonomy(self) -> None:
        """Take a run that was left going back, in the mode it was left under.

        A window opening onto a run that never stopped is the user who left it coming
        back, so attaching *is* the take-back: the claim that kept the process alive is
        dropped here, and the panel is told the server is attached. That is what makes
        closing this window end the run again — a server no conversation claims stops
        once its last client has been gone for the grace, and a claim nothing drops on
        arrival makes the process immortal, since a window that merely attached to a
        server holds no handle on it to kill.

        The level is a separate question, and the run's own answer is the truth: it is
        read before the claim goes and applied to the agents. Two things disagree with
        it otherwise — a webview arrives on its default, and a worker rebuilt during the
        absence comes up on the pool-wide record rather than the level chosen for this
        session, which would quietly drop a run from ``auto`` to ``manual`` and park it
        at its next sensitive call, with nothing said.

        Scoped to the run, not persisted past it: the level is read from the registry
        entry, which is cleared on a clean stop and liveness-checked. That is what keeps
        this from becoming the thing the approval mode is deliberately never persisted
        for — a mode that suppresses prompts being inherited by a later session that
        never asked for it.
        """
        held = {sid: lvl for sid, lvl in server_registry.claims().items()
                if lvl in ("manual", "auto", "auto_all")}
        if not held:
            return
        # Each conversation gets back the level *it* was left under. Applied to its own
        # agent, and recorded pool-wide only for the conversation on screen, which is
        # the one a worker built next most likely belongs to; every other claimed
        # conversation has its agent here already, since a claim is what kept it.
        for sid, lvl in held.items():
            worker = self.pool.get(sid)
            if worker is None:
                continue
            try:
                worker.apply_setting("approval_mode", lvl)
            except Exception:
                logger.warning("reattach: session %s refused the level %s it was left "
                               "under", sid, lvl, exc_info=True)
        level = held.get(self._active_session_id or "") or sorted(held.values())[-1]
        self._apply_setting("approval_mode", level)
        # Before anything is announced, because what the panel is about to be told —
        # that this window owns the server — has to be true of the registry the idle
        # sweep reads, and not only of the message. Every claim at once: one process
        # serves the whole workspace and this window is the only one looking at it, so
        # there is no conversation left for a claim to be held on behalf of.
        try:
            server_registry.unclaim(None)
        except Exception:
            # Reported and stepped over: the level below is the half the user can be
            # harmed by losing, and a claim that could not be dropped costs a process
            # that outlives this window — recoverable with the stop command.
            logger.warning("reattach: the detach claims could not be dropped",
                           exc_info=True)
        try:
            # That the server is this window's again, not only the level the run is
            # under. The panel decides from this what disconnecting costs, and it now
            # costs the runs: a window told otherwise would offer to walk away from
            # turns that end the moment it does.
            await self.ws.send(json.dumps({
                "type": "detached", "detached": False, "log": None,
                "autonomy": level, "sessions": [], "pid": os.getpid(),
                "setsid": False,
            }))
            await self.ws.send(json.dumps({"type": "approval_mode", "mode": level}))
            # Rendered, not notified: ``_notify`` writes to the transcript's transient
            # channel, which is dropped. Both halves of this are the kind of thing a
            # user walks back in needing to know — that the run has been approving
            # sensitive calls on its own, and that closing this window now ends it —
            # and neither is chatter.
            #
            # Quiet, though, and that is the point. It is not a warning: nothing has
            # gone wrong, nothing needs answering, and the user cannot act on it except
            # through controls already on screen. A badge and a coloured border read as
            # a problem to deal with on a conversation the user has just walked back
            # into — the one moment they are reading for what happened rather than for
            # what to do. So it reads as a rule across the thread, marking where they
            # rejoined and under what.
            #
            # Transient: it describes the state at this attachment, not anything that
            # happened in the conversation. Stored, every later load would replay it —
            # and a conversation rejoined twenty times would open on twenty of them,
            # each claiming to be now.
            #
            # It opens with the state the toggle shows — "attached" — because that is
            # the word the button answers to, and a line that only says what the run
            # has been doing leaves the user to work out which of the two positions
            # they are in. Then the consequence, which the glyph cannot show: the
            # process survived their absence and will not survive the next one unless
            # they ask again.
            #
            # The level is named only when something was deciding on its own, and the
            # count only when more than one conversation was left running: the levels
            # can differ, and "under auto_all" says nothing about which of them.
            under = f", under “{level}”" if level != "manual" else ""
            title = (
                f"attached — this conversation kept working with no window open"
                f"{under}; closing this window now stops it"
                if len(held) == 1 else
                f"attached — {len(held)} conversations kept working with no window "
                f"open, this one{under or ' at manual'}; closing this window now "
                f"stops them")
            await self._command_reply("detach", title, tone="quiet", transient=True)
        except Exception:
            return
        logger.info("reattach: %d claim(s) taken back; autonomy restored to %s from "
                    "the detached run", len(held), level)

    async def _send_replay(self) -> None:
        """Send what this session produced past the client's watermark, then go live.

        The order in :meth:`run` is the whole argument that this neither duplicates nor
        drops. The subscription was opened *before* the handshake, so everything the
        agents produced since is already queued behind it — there is no gap to fall
        into. The journal is then read from the watermark and sent. Events produced
        while that read was happening are in both places, so the subscription's gate is
        raised to where the replay ended and the drain loop discards anything at or
        below it: the overlap collapses to exactly one copy.

        Streamed deltas are not in the journal and so are not replayed — a resumed turn
        shows the aggregate that closed each block (``thinking_end``, ``answer``) rather
        than its keystrokes, which is also what makes replaying a twelve-hour run
        affordable. ``context_usage``, ``batch_status`` and ``todo`` are re-derived on
        load and are skipped for the same reason.
        """
        session_id = self._active_session_id
        if not session_id or self._sub is None:
            return
        # Never past what the journal actually holds. The watermark arrives from the
        # client, and is stored, so it can outlive the journal it counted — a session
        # whose log was removed, or a number inherited from another conversation. A
        # claim to have seen more than exists cannot be true, and honouring it sets the
        # gate above every event this session will ever produce.
        #
        # The gate it feeds is this conversation's own, which is what makes the clamp
        # enough: ``seq`` is counted per conversation, so a number from one says
        # nothing about another.
        held = self.pool.bus.last_seq(session_id)
        if self._rendered_seq > held:
            logger.warning("replay: session %s claims to have rendered seq %d but its "
                           "journal holds %d; trusting the journal",
                           session_id, self._rendered_seq, held)
            self._rendered_seq = held
        try:
            events, truncated = read_since(session_id, self._rendered_seq,
                                           limit=_REPLAY_MAX_EVENTS)
        except Exception:
            logger.warning("replay: session %s could not be read", session_id,
                           exc_info=True)
            return
        if not events:
            # Nothing missed. The gate still moves to where the journal stands, so a
            # live event this connection has already been handed is not sent twice.
            self._sub.rendered_through(session_id, self._rendered_seq)
            return

        through = self._rendered_seq
        for start in range(0, len(events), _REPLAY_CHUNK):
            chunk = events[start:start + _REPLAY_CHUNK]
            through = max(through, max(int(e.get("seq", 0)) for e in chunk))
            try:
                await self.ws.send(json.dumps({
                    "type": "replay",
                    "session_id": session_id,
                    "events": chunk,
                    "through_seq": through,
                    "more": start + _REPLAY_CHUNK < len(events),
                    "truncated": truncated and start == 0,
                }, default=str))
            except Exception:
                return
        # Only now: a gate raised before the frames were sent would have discarded the
        # live events that arrived between the read and the send.
        self._sub.rendered_through(session_id, through)
        logger.info("replay: session %s resent %d event(s) from seq %d",
                    session_id, len(events), self._rendered_seq)

    async def _notify(self, text: str) -> None:
        """One line of chrome for the client, outside any conversation. Never raises."""
        try:
            await self.ws.send(json.dumps({"type": "output", "text": f"  {text}\n"}))
        except Exception:
            pass

    async def _compact_history(self) -> bool:
        """Summarize the middle of the session history. True when it actually shrank.

        Keeps the opening user message and the last two exchanges and replaces what
        lies between with a single summary — the same shape the agent loop's
        intra-query compaction uses. The summarization is an LLM call, so it happens
        on the worker thread (``_AgentWorker.compact_middle``); this coroutine only
        awaits its future. ``history_full`` is deliberately untouched: it stays the
        one complete account of the session, exactly as for a front-trim.

        Never raises — the caller falls back to front-trimming on a False.
        """
        history = self.conv.history
        middle = history[1:-4]
        if len(middle) < 3:
            return False
        await self.ws.send(json.dumps({"type": "output",
            "text": "  ⚡ Context budget reached — compacting older history…\n"}))
        try:
            summary = await asyncio.wrap_future(self.worker.compact_middle(list(middle)))
        except Exception:
            return False
        # compact_messages returns its input unchanged when summarization failed.
        if not summary or len(summary) >= len(middle):
            return False
        history[1:-4] = summary
        # Slicing across a turn boundary can strand an assistant tool call or its
        # result; strict backends reject that outright.
        history[:] = reconcile_tool_pairs(history)
        self.pool.bus.record_client_event(self._active_session_id, {
            "type": "context_compact",
            "dropped": len(middle),
            "kept": len(history),
            "archived": len(self.conv.history_full),
        })
        return True

    def _apply_setting(self, name: str, value: Any) -> str:
        """Apply a UI knob to every live conversation, and to the next agent built.

        Both halves are needed. Pushing it to the live workers makes the change take effect
        now; recording it in the pool makes a conversation started afterwards come up on it
        rather than on the default, which matters because an agent is built long after the
        user sets these.

        Returns the first rejection a live worker reported, or "". Nothing to reject when
        none exists yet: the setting is recorded and validated when it is replayed.
        """
        return next(iter(self.pool.apply_setting(name, value)), "")

    async def _ensure_worker(self) -> _AgentWorker | None:
        """This conversation's agent, built now if it has none and there is room.

        None means the pool is full of conversations that are working, waiting on the user,
        or watching a job — none of which may be evicted to make room. The caller queues
        instead, and the user is told, because a queue nobody can see reads as a hang.

        The build costs the LLM backend wait plus ~19 MCP server spawns, so it runs off the
        event loop — the other conversations keep streaming — and this one announces it.
        """
        session_id = self._active_session_id
        if not session_id:
            return None
        existing = self.pool.get(session_id)
        if existing is not None:
            return existing

        def _announce() -> None:
            self._schedule(self.ws.send(json.dumps({
                "type": "status",
                "session_id": session_id,
                "text": "  ⏳ Starting an agent for this conversation…",
            })))

        try:
            worker = await self.pool.worker_for(session_id, on_wait=_announce)
            if worker is not None:
                worker.session_title = self._session_title(session_id)
            return worker
        except Exception as exc:
            await self.ws.send(json.dumps({
                "type": "error",
                "session_id": session_id,
                "text": f"Could not start an agent for this conversation: {exc}",
            }))
            return None

    def _schedule(self, coro: Any) -> None:
        """Fire a send without awaiting it, for callers that are not coroutines."""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()

    async def _fit_history_to_budget(self) -> None:
        """Make the window fit the budget before a new query is added to it.

        Compaction first, front-trim as the fallback, and both are real losses of
        context — so this must run against the budget the conversation actually has.
        Its caller resolves the agent first for that reason.
        """
        # Pre-query budget check: front-trim the oldest history so the new query fits.
        # Deliberately not compact_history — an LLM call from the event loop is unsafe
        # while the worker thread owns the agent.
        total, reserved = self._ctx_budget()
        # Subtract the per-call overhead (system prompt + tools schema) that rides on
        # every call: without it the bar reads past 100% while this check says it fits.
        usable_tokens = max(1, total - reserved - self.worker.context_overhead_tokens())
        # allow_network=False: never block the WS event loop on tokenize.
        backend = get_backend()
        history = self.conv.history
        counts = backend.message_token_counts(
            self.worker.model, history, allow_network=False
        )
        used_tokens = sum(counts)
        if used_tokens >= usable_tokens and len(history) > 1:
            # Compaction first: it carries forward what the older turns established,
            # where the front-trim below just forgets them. Only when it cannot run
            # (too short a middle) or the summary still doesn't fit do we drop turns.
            if await self._compact_history():
                counts = backend.message_token_counts(
                    self.worker.model, history, allow_network=False
                )
                used_tokens = sum(counts)
                await self._emit_context_usage()
        if used_tokens >= usable_tokens and len(history) > 1:
            await self.ws.send(json.dumps({"type": "output",
                "text": "  ⚡ Context budget reached — trimming oldest history…\n"}))
            # Per-message counts computed once; decrement as we pop the front.
            dropped_before = len(history)
            total_tokens = used_tokens
            idx = 0
            while total_tokens > usable_tokens and len(history) > 1 and idx < len(counts):
                history.pop(0)
                total_tokens -= counts[idx]
                idx += 1
            # Front-trimming can orphan a ``{"role": "tool"}`` whose assistant tool_call
            # was popped, which a strict tokenizer rejects — so drop leading
            # tool messages until history starts on a valid turn boundary.
            while history and history[0].get("role") == "tool":
                history.pop(0)
            # `history_full` is deliberately untouched above — this records what the
            # window lost, so the log stays the one complete account of the session.
            self.pool.bus.record_client_event(self._active_session_id, {
                "type": "context_trim",
                "dropped": dropped_before - len(history),
                "kept": len(history),
                "archived": len(self.conv.history_full),
            })
            await self._emit_context_usage()

    async def _handle_query(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return
        # Writing instead of answering moves past a deferred card: the call keeps its
        # "not run" result, and a late answer to the card goes nowhere.
        if self.conv.pending_interaction is not None:
            prompt_id = (self.conv.pending_interaction.get("prompt") or {}).get("id")
            if prompt_id:
                self._stale_prompt_ids.add(prompt_id)
            self.conv.pending_interaction = None

        # Resolve @<uri> mentions on the worker's loop (where the MCP sessions live),
        # then submit the augmented text. History and display keep the RAW `text`, so
        # the attachment is per-turn. Caveat: in full-context mode the augmented message
        # persists in the worker's `_last_full_messages`.
        effective_text = text
        # Built here, before anything is submitted: resolving @-mentions needs the
        # worker's own loop, where the MCP sessions live. None means the pool is full,
        # and the turn is queued below with the mentions left unresolved — the text is
        # still exactly what the user wrote.
        worker = await self._ensure_worker()
        # After the agent, never before it: the budget is sized from its context mode
        # and its prompt overhead, and neither can be read off a conversation that has
        # no agent yet. A resumed session asked too early was budgeted at compact's 32k,
        # which read as an overflow and compacted and trimmed a history that fit the
        # real window perfectly well.
        await self._fit_history_to_budget()
        attached_uris: list[str] = []
        if worker is not None:
            try:
                effective_text, attached_uris = await asyncio.wrap_future(
                    worker.resolve_resources(text)
                )
            except Exception:
                attached_uris = []
        if attached_uris:
            await self.ws.send(json.dumps({
                "type": "output",
                "text": "📎 Attached: " + ", ".join(attached_uris) + "\n",
            }))

        self.conv.add_user_message(text, show=True)
        self.pool.bus.record_client_event(
            self._active_session_id, {"type": "query", "text": text})
        # Save on arrival so a reconnect mid-turn (e.g. while waiting for an edit
        # approval) reloads from disk instead of minting a blank session ID, which
        # would wipe the chat on the frontend.
        self._save()
        messages = (list(self.conv.history[:-1])
                    + [{"role": "user", "content": effective_text}])
        session_id = self._active_session_id

        def _submit(worker: _AgentWorker) -> None:
            worker.submit_query(effective_text, messages, session_id=session_id)

        if worker is not None:
            _submit(worker)
            return
        # Every slot is held by a conversation that is working, waiting on the user, or
        # watching a job. Wait for one rather than evicting: taking a slot from a
        # conversation mid-task trades a visible wait for silently lost work.
        position = self.pool.enqueue(session_id, _submit)
        await self.ws.send(json.dumps({
            "type": "queued",
            "session_id": session_id,
            "position": position,
            "text": (f"  ⏸ Waiting for a free agent slot (#{position} in line). "
                     f"At most {self.pool.cap} conversations run at once."),
        }))

    @staticmethod
    def _text_count(messages: list) -> int:
        """Number of plain-text bubbles — the part of a transcript both sides share.

        A note this layer wrote itself (``role: "system"``) is not that part: the
        webview never received it, so counting it made every later transcript the
        client sent look short by one and the guard below refused them all. The stored
        chat then froze at the moment of the first notice and came back stripped of
        every tool row, diff card and reasoning panel after it.
        """
        return sum(1 for m in messages
                   if isinstance(m, dict) and m.get("kind", "text") == "text"
                   and m.get("role") != "system")

    async def _handle_transcript(self, msg: dict) -> None:
        """Store the client's rendered transcript as this session's display messages.

        The rich transcript — tool rows, reasoning panels, diff cards — is assembled by
        the webview's reducer and exists nowhere else; the server only ever appended the
        text bubbles, which is why a reload used to come back stripped to prose. Rather
        than rebuild that assembly here, we take the client's copy of it.

        Two things make that safe to trust: the transcript must name the session it
        belongs to (one arriving after a switch would otherwise overwrite the session the
        user just moved to), and it must not have fewer text bubbles than what we hold —
        a webview that just opened on an empty view must never blank a stored history.
        """
        if self._active_session_id is None:
            return
        if msg.get("session_id") != self._active_session_id:
            return
        sid = self._active_session_id
        # Per session: the client's copy as last applied, by id, and which revision of
        # which run of full-then-deltas it is. A delta is only ever applied to that.
        syncs = getattr(self, "_transcript_sync", None)
        if syncs is None:
            syncs = self._transcript_sync = {}
        if msg.get("delta"):
            held = syncs.get(sid)
            order = msg.get("order")
            changed = msg.get("changed")
            if (held is None or held["epoch"] != msg.get("epoch")
                    or held["rev"] != msg.get("base_rev")
                    or not isinstance(order, list) or not isinstance(changed, list)):
                await self._send_transcript_resync(sid)
                return
            by_id = dict(held["by_id"])
            for m in changed:
                if isinstance(m, dict) and m.get("id"):
                    by_id[m["id"]] = m
            if any(i not in by_id for i in order):
                await self._send_transcript_resync(sid)
                return
            messages = [by_id[i] for i in order]
        else:
            messages = msg.get("messages")
            if not isinstance(messages, list):
                return
        if self._text_count(messages) < self._text_count(self.conv.display):
            return
        syncs[sid] = {
            "epoch": msg.get("epoch"),
            "rev": msg.get("rev", 0),
            "by_id": {m["id"]: m for m in messages if isinstance(m, dict) and m.get("id")},
        }
        self.conv.display = messages
        # The client says how far the transcript it just sent reaches. Taken on trust
        # the same way the transcript itself is, and only ever forward: a webview that
        # reopened on less than we hold must not shorten the watermark either.
        through = msg.get("through_seq")
        if isinstance(through, int) and through > self._rendered_seq:
            self._rendered_seq = through
        self._save()

    async def _send_transcript_resync(self, session_id: str) -> None:
        """Ask for the transcript whole: a delta arrived on a revision not held here."""
        try:
            await self.ws.send(json.dumps(
                {"type": "transcript_resync", "session_id": session_id}))
        except Exception:
            pass

    async def _handle_steer(self, msg: dict) -> None:
        """A message typed while the agent is busy — inject it into the running run.

        The message is recorded in history/display (so it persists like a normal
        turn) and handed to the worker's steer queue, which the agent loop drains at
        its next step boundary. If no run is actually in flight (race: the agent just
        finished), fall back to the normal query path so the message isn't dropped.
        """
        text = (msg.get("text") or "").strip()
        if not text:
            return
        if not self._running_turn_is_ours():
            # Nothing of ours in flight — or a detached wake turn running in another
            # session, which this message must not be injected into. Either way the
            # normal query path is right: the serial query loop queues it.
            #
            # The client sent a steer because it still believes a run is going, and the
            # end of a run reaches the worker before its answer reaches the screen: a
            # message typed in that gap is answered as a turn of its own, so it is also
            # told so — unannounced, the bubble kept the "queued" tag of a run that was
            # already over.
            await self._announce_steer_turn(text)
            await self._handle_query(msg)
            return
        self.conv.add_steer(text)
        self.pool.bus.record_client_event(
            self._active_session_id, {"type": "steer", "text": text})
        self._save()
        self.worker.submit_steer(text)

    async def _announce_steer_turn(self, text: str) -> None:
        """Tell the client the steer it just sent became a turn of its own.

        The same ``steer_injected`` :meth:`_announce_steer_turns` sends, and for the
        same two reasons: the bubble drops the "queued" tag of a run that is over, and
        ``busy`` is set for a turn the user is waiting on but never pressed send for.

        It goes on the agent's output queue rather than to the socket, because what
        makes it read correctly is where it sits relative to the answer of the run the
        message was typed into. That answer is on that queue, ahead of this, and it is
        what clears ``busy``: sent straight to the socket this would arrive first and
        the turn would run with no stop button and no end the composer can see. With no
        agent there is no queue and no answer in flight either, so nothing to be ordered
        behind — the socket is then the only route.
        """
        ev = {"type": "steer_injected", "text": text, "starts_turn": True,
              # Unstamped it would pass the foreign-event filter and clear a tag in
              # whatever conversation happens to be on screen.
              "session_id": self._active_session_id}
        out_q = getattr(self.pool.get(self._active_session_id), "out_q", None)
        if out_q is not None:
            out_q.put(ev)
            return
        try:
            await self.ws.send(json.dumps(ev))
        except Exception:
            pass

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

    def _rows_for(self, session_id: str | None) -> dict[str, str]:
        """The divertible tool rows of one conversation's running turn."""
        return self._live_rows.setdefault(session_id or "", {})

    def _forget_row(self, call_id: str, session_id: str | None = None) -> None:
        """Drop a finished row from the divert/progress bookkeeping."""
        sid = session_id if session_id is not None else self._active_session_id
        self._rows_for(sid).pop(call_id, None)
        self._sent_progress.setdefault(sid or "", {}).pop(call_id, None)

    async def _handle_shutdown(self, msg: dict) -> None:
        """Stop the server, now or once it has nothing left to do.

        ``force`` stops it whatever is in flight — the user's own decision about their
        own machine, and the only way to end a run that is waiting on an answer they
        have decided not to give. Without it the request is refused while anything is
        still working, and the reasons are sent back: an arrest that silently discarded
        a two-hour build would be the worst answer this path could give.

        Either way it goes through the same exit as a signal, which is the only thing
        that closes the agents' MCP servers.
        """
        report = self.pool.idle_report()
        force = bool(msg.get("force"))
        if not report["idle"] and not force:
            try:
                await self.ws.send(json.dumps({
                    "type": "shutdown_refused",
                    "reasons": report["reasons"],
                }))
            except Exception:
                pass
            return
        try:
            await self.ws.send(json.dumps({
                "type": "shutting_down",
                "forced": force,
                "reasons": report["reasons"],
            }))
        except Exception:
            pass
        logger.info("shutdown: requested by the client (forced=%s; %s)", force,
                    "; ".join(report["reasons"]) or "nothing was running")
        self.pool.request_stop()

    def _safe_list_sessions(self) -> list:
        """The stored conversations, or none when the directory cannot be read.

        Guarded because its one caller is deciding what to claim: a listing that
        raises must not turn a detachment into an error, and an unreadable store is
        the one case where "this names no conversation" cannot be concluded — so it
        falls back to the agents, which is the stricter answer.
        """
        try:
            return self.store.list_sessions()
        except Exception:
            logger.warning("detach: the stored conversations could not be listed",
                           exc_info=True)
            return []

    async def _handle_detach(self, msg: dict) -> None:
        """Let the server go on without this window — "continue without me".

        Served here, on the WS loop, because the two things that make this process
        survivable are things it does to *itself*: re-pointing its output fds away from
        the pipe the extension host holds, and leaving that host's process group. The
        spawn is not touched at all, which is why this works on a server that was
        started as an ordinary child — and why it can be decided when the user is
        leaving rather than when they connected, which is the moment they actually know
        whether anything is worth leaving running.

        The autonomy level is the user's answer to "what may it do while I am gone", and
        it is required rather than defaulted: a detachment at ``manual`` parks at the
        first sensitive tool and does almost nothing overnight, and that has to be a
        choice rather than something that happened to them. It is applied to the
        conversations named, through the same seam ``/approvals`` uses, so a turn
        already in flight picks the new level up at its next gate.

        Naming no conversation means all of them, and then the level is recorded
        pool-wide as well — which is the only form that survives a worker being rebuilt,
        since the pool's record of UI settings is per-pool and not per-session.
        """
        # Giving the decision back. The process-level work cannot be undone — fds that
        # point at a log file have no pipe to return to — but none of it needs undoing:
        # what makes a server survive a window closing is that nobody kills it, and that
        # is a decision, not a state of the process. So this clears the claim and leaves
        # the redirect in place.
        if msg.get("enabled") is False:
            # Per conversation, like the claim it undoes: naming some takes back only
            # those, naming none takes back every one. The process goes on surviving
            # for as long as any claim stands — it is one process, and the redirect
            # that makes it survivable belongs to it rather than to a conversation.
            named = msg.get("session_ids")
            held = server_registry.unclaim(
                [str(s) for s in named] if isinstance(named, list) and named else None)
            # Read before the send, not inside it: a getter that raises would otherwise
            # swallow the whole reply, and a client left believing the server is still
            # detached stops guarding something nothing is guarding.
            try:
                level = self.pool.worker_or_detached(
                    self._active_session_id).setting("approval_mode")
            except Exception:
                level = "manual"
            try:
                await self.ws.send(json.dumps({
                    "type": "detached", "detached": bool(held), "log": None,
                    "autonomy": level, "sessions": sorted(held), "pid": os.getpid(),
                    "setsid": False,
                }))
                # The switch itself, in the same quiet rule the attachment notice
                # uses. The button is one glyph and both of its positions look like a
                # chain: a line in the thread is what tells the user which way it just
                # went, and it says the consequence rather than only the word, since
                # "attached" on its own does not warn anybody that closing the window
                # now ends the run.
                await self._command_reply(
                    "detach",
                    ("attached — this window owns the server again, and closing it "
                     "stops the run" if not held else
                     f"attached — this window owns the server again; {len(held)} "
                     f"other conversation(s) still keep it alive"),
                    tone="quiet", transient=True)
            except Exception:
                return
            logger.info("detach: taken back for %s; %d claim(s) left",
                        ", ".join(str(s) for s in named) if isinstance(named, list)
                        and named else "every conversation", len(held))
            return

        autonomy = str(msg.get("autonomy") or "manual")
        if autonomy not in ("manual", "auto", "auto_all"):
            try:
                await self.ws.send(json.dumps({
                    "type": "error",
                    "text": (f"Unknown autonomy level: {autonomy}. "
                             "Use manual, auto, or auto_all."),
                }))
            except Exception:
                pass
            return

        # A claim is a *conversation's* request to be left running, so the set is the
        # conversations named — never the live agents. An agent is built on a
        # conversation's first query and released after ten minutes of idleness, and
        # the ones with no agent at that moment are exactly the ones this exists for: a
        # conversation whose work is a job MIMIR cannot see has nothing holding its
        # agent, so it is asleep when the user leaves. Built from the workers, the set
        # came out empty for it — and an empty claim is written as "not detached", so
        # the server stopped thirty seconds after the window closed, having just been
        # asked to keep going.
        #
        # The *level* still needs an agent to be applied to, and that is a separate
        # thing: one asleep gets it back from the claim when a window reattaches, and
        # from the pool-wide record when something rebuilds it meanwhile.
        session_ids = msg.get("session_ids")
        targeted: list[str] = []
        if isinstance(session_ids, list) and session_ids:
            known = {meta.id for meta in self._safe_list_sessions()}
            for sid in session_ids:
                worker = self.pool.get(sid)
                if worker is None and str(sid) not in known:
                    # Not asleep — not a conversation at all. Claiming an id nothing
                    # knows about would hold this process open for something that
                    # cannot come back to take the claim off it.
                    logger.info("detach: ignoring %s, which names no conversation", sid)
                    continue
                targeted.append(str(sid))
                if worker is None:
                    continue
                try:
                    worker.apply_setting("approval_mode", autonomy)
                except Exception:
                    logger.warning("detach: session %s refused the autonomy level %s",
                                   sid, autonomy, exc_info=True)
        else:
            self._apply_setting("approval_mode", autonomy)
            # Naming none means all of them, and "all" includes the conversation on
            # screen even before it has asked anything: a window detached the moment it
            # opened claimed nothing at all, because no agent had been built yet.
            targeted = [sid for sid, _w in self.pool.items()]
            if self._active_session_id and self._active_session_id not in targeted:
                targeted.append(self._active_session_id)

        # The agent's stdout is about to become a log file, which is not a tty but is
        # also not proof that nobody is reachable. Saying so plainly keeps the
        # interactive fallbacks — the bare ``input()`` calls behind the CLI's hooks —
        # from being attempted at all.
        #
        # Through the pool, so it is recorded as well as pushed: a detached process
        # builds agents after this point — a conversation whose own agent was released
        # gets it back when a run of its finishes — and the redirect cannot be undone,
        # so one that came up without this would reach for a terminal that is a log
        # file for the rest of the process's life.
        self.pool.set_non_interactive()

        info = detach_process(_MIMIR_DIR_WS)
        # The entry already holds the address; this only adds what has changed about
        # the process behind it, so a window that attaches later knows which
        # conversations were left running and under which level each.
        #
        # Per conversation, because that is the unit the user chose. The *process* is
        # not: one serves the whole workspace, and the redirect that makes it
        # survivable is its own — so it lives for as long as any conversation claims
        # it, and becomes mortal again when the last claim goes.
        server_registry.claim(targeted, autonomy, log=info.get("log"))

        try:
            await self.ws.send(json.dumps({
                "type": "detached",
                "detached": True,
                "log": info.get("log"),
                "autonomy": autonomy,
                "sessions": targeted,
                "pid": info.get("pid"),
                "setsid": info.get("setsid"),
            }))
            # The other half of the switch — see the take-back above. Named here as
            # what the run may do while unwatched, because that is the part of
            # detaching the user chose and the part the glyph cannot show.
            #
            # And checked against what this process would actually still have to do
            # once this window has gone, because "this run keeps working" is a claim
            # about that and nothing was testing it. A conversation whose work is a
            # job MIMIR never submitted — a raw ``sbatch`` in a shell, a submission
            # over ``ssh`` — has nothing on disk for the scan to find, so the process
            # is idle by every measure it has and stops on the idle TTL a couple of
            # hours later. Detaching into that is a night lost to a sentence that was
            # never true, and the moment to say so is the click.
            idle = self.pool.idle_report(excluding=self._sub)
            if idle["idle"]:
                # Conditional, and that is not hedging: this is the state at the click,
                # and the click can land while a turn is one tool call away from
                # submitting the very job that would keep the process alive. Said
                # flatly — "nothing is running, this will stop" — the line would be
                # false by the time it was read, which is how a true warning becomes
                # one the user learns to skip. So it says what is so now, and what
                # follows only if nothing starts. The clock itself is described by the
                # one author of that sentence, next to the TTL it is measured against.
                title = f"detached under “{autonomy}” — nothing is being watched yet"
                items = [{"label": "autonomy", "detail": autonomy}]
                note = ("No turn in flight and no run MIMIR is watching at this "
                        "moment. "
                        + idle_stop_reason(self.pool.server_idle_ttl, past=False)
                        + " Work it did not start itself is not watched: ask it to "
                        "watch the job or the process before you go.")
            else:
                title = (f"detached — this run keeps working with this window closed, "
                         f"under “{autonomy}”" if autonomy != "manual" else
                         "detached — this run keeps working with this window closed, "
                         "and parks at the first call that needs you")
                items = [{"label": "keeping it going", "detail": reason}
                         for reason in idle["reasons"]]
                note = ""
            await self._command_reply("detach", title, items=items, note=note,
                                      tone="quiet", transient=True)
        except Exception:
            return
        logger.info("detach: this server is now survivable (autonomy %s over %d "
                    "conversation(s))", autonomy, len(targeted))

    async def _handle_divert_to_background(self, msg: dict) -> None:
        """Move the run the user pointed at into the background.

        Served here, on the WS loop, and deliberately never through the model: the
        worker thread is parked awaiting the tool result for the whole call, and the
        steer queue is drained only at a step boundary, so an instruction routed that
        way could not arrive until after the run it was meant to divert had ended.
        Writing the request to the shared state dir is what reaches a server whose
        event loop that very call is holding.

        Which run is decided by the row the user clicked: the message carries the call
        id, and the row's tool name is the channel the owning server publishes under.
        Resolved here rather than shipped by the webview, which has no business
        knowing a tool name. Sending the request to the shell's channel whatever was
        clicked has a proxy run report "nothing to move" while a perfectly innocent shell
        command is the one that gets detached.

        Nothing worker- or agent-side is touched: the confirmation the user sees is
        the tool result that lands a moment later, carrying the work done so far and
        the job handle. Saying anything more here would be predicting it.
        """
        channel = self._rows_for(self._active_session_id).get(str(msg.get("id") or ""))
        if channel is None or run_channel.request_divert(
                channel, self._active_session_id) is None:
            await self.ws.send(json.dumps({
                "type": "status",
                "text": "  ⓘ Nothing to move — that run had already finished.",
            }))

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
            worker.resolve_approval(answer["choice"], answer["approved_files"],
                                    str(msg.get("id") or ""))

    async def _handle_user_question_response(self, msg: dict) -> None:
        if await self._answer_deferred(msg, {"answers": msg.get("answers") or []}):
            return
        worker = self._worker_for_answer(msg)
        if worker is not None:
            # With the card's id: an answer reaches the card it was shown on, and one
            # naming a card that is gone settles nothing.
            worker.resolve_question(msg.get("answers"), str(msg.get("id") or ""))

    async def _resend_deferred_prompt(self) -> None:
        """Put the card a deferred turn of this session waits on back on screen."""
        prompt = (self.conv.pending_interaction or {}).get("prompt")
        if not prompt:
            return
        try:
            await self.ws.send(json.dumps(prompt, default=str))
        except Exception:
            pass

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

    async def _admit_waiting(self) -> None:
        """Start the turn of a conversation that was waiting for an agent slot.

        Called the moment a turn ends, because that is when its conversation stops being
        busy and so becomes the one the pool may release to make room. Leaving it to the
        idle sweep makes a queued conversation wait out the sweep interval after the slot
        it needs has already freed — half a minute of nothing, having just been told it is
        next in line.
        """
        try:
            for session_id in await self.pool.pump():
                await self.ws.send(json.dumps({
                    "type": "status",
                    "session_id": session_id,
                    "text": "  ▶ An agent slot freed — this conversation is starting.",
                }))
        except Exception:
            logger.warning("could not admit a queued conversation", exc_info=True)

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
            pass

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

    async def _handle_clear_todos(self, msg: dict) -> None:
        self.worker._clear_todos()
        await self.ws.send(json.dumps({"type": "todo", "items": []}))

    def _running_turn_is_ours(self) -> bool:
        """True when the conversation on screen has a turn in flight.

        Asked of that conversation's own agent. Several can be working at once, so what
        matters for a message the user typed is whether *this* conversation is busy, in
        which case the message steers its turn instead of starting one.
        """
        return self.pool.is_busy(self._active_session_id)

    def _is_foreign_event(self, ev: dict) -> bool:
        """True when *ev* was produced for a session other than the active one.

        Every event of a running turn is stamped with the session it was produced for.
        Several conversations can be producing at once, and only one of them is on screen,
        so the stamp is what keeps one conversation's output out of another's chat — it goes
        to that conversation's own transcript instead. Unstamped events (produced outside
        any conversation) always pass.

        So do the interaction events. Each one is a question the agent is parked on,
        and a background-job wake runs turns in sessions the user is not looking at:
        filtering those prompts as foreign would park the turn forever on an answer
        nobody was shown. They carry their session id, so the prompt can say which
        conversation is asking.
        """
        if ev.get("type") in _INTERACTION_EVENTS:
            return False
        ev_session = ev.get("session_id")
        return ev_session is not None and ev_session != self._active_session_id

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
        Slurm last gave it — ``job_scan.is_settled`` holds both halves of that question,
        because a Slurm job records no exit code and asking only for one reads every job
        a conversation ever submitted as still running. Neither is a certainty, which is
        why this reports and does not act.
        """
        base = os.path.join(_MIMIR_DIR_WS, "sessions", session_id)
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

    async def _handle_command_msg(self, msg: dict) -> None:
        await self._handle_command((msg.get("text") or "").strip())

    async def _command_reply(
        self,
        command: str,
        title: str,
        *,
        items: list[dict] | None = None,
        note: str = "",
        tone: str = "ok",
        transient: bool = False,
    ) -> None:
        """Send one structured answer for a session command.

        Only two kinds of command reach here: a LISTING, and something IRREVERSIBLE
        that just happened. Every setting — mode, thinking, streaming, batch,
        context, approvals, enforcement — reports state instead and writes nothing to
        the transcript: each already has a control that shows its value, so a line
        about it repeats the chrome. The webview also replays its stored settings on
        every connect, so those lines were not even reports of anything the user had
        just done; they greeted each session with changes nobody had made.

        Structured, not pre-formatted. A listing of twenty memories and a one-line
        result want different shapes on screen, and a frontend cannot lay out what
        reaches it as an already-indented blob of text: the best any frontend can do with
        a blob is print it.

        ``tone`` says how the result should read: ``ok`` for routine, ``warn`` for
        something irreversible that just happened, ``empty`` for a listing with
        nothing in it, ``quiet`` for a state the user has walked back into rather than
        caused — drawn as a rule across the thread, with no badge and no border, since
        there is nothing to deal with. ``items`` are ``{label, detail}`` rows.

        ``transient`` keeps the card out of the stored transcript. For one that
        describes this moment rather than the conversation: stored, it is replayed on
        every load, so a conversation reconnected to twenty times reopens on twenty
        copies of it, each above whatever answer it happened to follow.
        """
        await self.ws.send(json.dumps({
            "type":    "command_output",
            "command": command,
            "title":   title,
            "items":   items or [],
            "note":    note,
            "tone":    tone,
            "transient": transient,
        }))

    async def _handle_command(self, text: str) -> None:
        """Answer a session command the user typed (never seen by the model)."""
        result = await WS_COMMANDS.run(WsCommandHost(self), text)
        if result is None:
            result = CommandResult.error(f"Unknown command: {text}")
        await self._send_command_result(result)

    async def _send_command_result(self, result: CommandResult) -> None:
        """A command's result, the way this front-end shows each shape.

        An error as ``error``. A setting as its STATE message — the webview draws it on
        the control that owns the setting, and a line in the transcript would only
        repeat that control. A listing or an irreversible change as ``command_output``,
        not ``output``: ``output`` is the transient tool-activity channel the webview
        drops on purpose, which silently swallowed every answer — ``/memory list``
        printed nothing, and ``/memory clear`` wiped the store while looking like it
        had done nothing at all.
        """
        if result.failed:
            await self.ws.send(json.dumps({"type": "error", "text": result.title}))
        elif result.state is not None:
            await self.ws.send(json.dumps(result.state))
        elif result.card:
            await self._command_reply(
                result.label, result.title,
                items=[{"label": i.label, "detail": i.detail} for i in result.items],
                note=result.note, tone=result.tone)

    # ── server / skill toggles ─────────────────────────────────────────────────
    async def _send_toggles(self) -> None:
        """Push the current server/skill enabled state for the toggle panel."""
        state = self.worker.toggles_state()
        await self.ws.send(json.dumps({
            "type": "toggles_list",
            "servers": state.get("servers", []),
            "skills": state.get("skills", []),
            "nudges": state.get("nudges", []),
        }))

    async def _handle_list_toggles(self, msg: dict) -> None:
        await self._send_toggles()

    async def _handle_list_panel(self, msg: dict) -> None:
        """Serve the scientific-computing panel: the servers' sections, plus ours.

        One section is the client's own because its state lives here — the runs the
        watcher is holding. The rest is whatever the connected servers declared they
        could fill.

        Two flags, and both are the difference between the reader asking and a timer
        asking. ``build`` starts this conversation's agent if it has none: the sections
        are filled by MCP servers, a conversation holds none until its first query, and
        a drawer opened before that query showed nothing — which is why the clusters
        only ever appeared once something had been asked of the model. ``refresh`` goes
        on to the servers as "read it again", past the caches they answer a timer from.

        The build is started, never waited on. It is the LLM backend wait plus ~19
        server spawns, and this handler runs on the socket's own read loop — awaiting it
        here would stop the user from typing for as long as it takes, for a drawer. So
        the report below goes out now, saying no agent answered for it; the worker
        announces itself with a ``ready`` once it is up, and that is the drawer's cue to
        ask again — and this end sends the report itself as soon as the build lands, so
        the drawer fills whether or not that cue makes it back.

        The report is not awaited here either. A cold cluster section is seconds of
        ``squeue``/``sinfo``, and the read loop held for it kept the user's own messages
        waiting behind a drawer. Asks that arrive while one is being served are folded
        into a single further pass, so a timer and a ⟳ never queue up reports.
        """
        if msg.get("build") and self.pool.get(self._active_session_id) is None:
            self._schedule(self._report_panel_once_built(self._ensure_worker()))
        self._request_panel(refresh=bool(msg.get("refresh")))

    def _request_panel(self, *, refresh: bool = False) -> None:
        """Have a panel report sent, without waiting for it."""
        wanted = self._panel_again or {}
        self._panel_again = {"refresh": wanted.get("refresh", False) or refresh}
        if self._panel_task is None or self._panel_task.done():
            self._panel_task = asyncio.get_running_loop().create_task(
                self._serve_panel())

    async def _report_panel_once_built(self, build: Any) -> None:
        if await build is not None:
            self._request_panel()

    async def _serve_panel(self) -> None:
        while self._panel_again is not None:
            msg, self._panel_again = self._panel_again, None
            try:
                await self._send_panel_report(refresh=msg["refresh"])
            except Exception:
                logger.debug("panel report failed", exc_info=True)
                return

    async def _send_panel_report(self, *, refresh: bool) -> None:
        # One worker for the whole report: the agent may land during the await, and a
        # report read off the stand-in must not then claim to be the agent's.
        worker = self.worker
        sections = await asyncio.wrap_future(worker.panel_sections(force=refresh))
        agent_ready = worker.agent_ready()
        if not agent_ready:
            from ...config import paths
            sections = recorded_hosts_sections(paths.STATE_DIR)
        # The one place both halves exist: what a cluster is made of came from a
        # server, what is running on it is this client's own watcher state. A run that
        # no group claimed still travels, under `runs`, for the flat list — a run the
        # user cannot see is worse than one shown out of place.
        sections, unplaced = attach_runs(sections, self.worker.watched_runs())
        cards = _list_subagents(self._active_session_id or "")
        await self.ws.send(json.dumps({
            "type": "panel_report",
            # Whether there was anything to ask. An empty section list from a
            # conversation with no agent is not "no clusters here".
            "agent_ready": agent_ready,
            "subagent_level": self.worker.setting("subagent_level"),
            "subagents": cards,
            "running": sum(1 for c in cards if c.get("state") == "running"),
            "runs": unplaced,
            "sections": sections,
        }))

    async def _handle_list_subagents(self, msg: dict) -> None:
        """Serve the sub-agents of the session in view.

        Read-only, and read off the files the children left: a panel that could relaunch
        one would be a second way to delegate, out of the orchestrator's sight.
        """
        cards = _list_subagents(self._active_session_id or "")
        await self.ws.send(json.dumps({
            "type": "subagents_list",
            "session_id": self._active_session_id or "",
            "subagents": cards,
            # For the badge on the button: how many are working right now.
            "running": sum(1 for c in cards if c.get("state") == "running"),
        }))

    async def _handle_read_subagent(self, msg: dict) -> None:
        await self.ws.send(json.dumps({
            "type": "subagent",
            "subagent": _read_subagent(
                self._active_session_id or "", str(msg.get("id") or "")),
        }))

    async def _handle_list_resources(self, msg: dict) -> None:
        """Serve the attachable-resource list for the webview picker/autocomplete."""
        await self.ws.send(json.dumps({
            "type": "resources",
            "resources": self.worker.resources_snapshot(),
        }))

    async def _handle_toggle_server(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("server", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_skill(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("skill", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_nudge(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.pool.apply_toggle("nudge", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _send_served_models(self) -> None:
        """Tell the client what the endpoint serves, so it can offer a choice.

        The agent process is the one authority on this: it holds the address and the
        API key, and it reaches the cluster the way the rest of the client does. The
        VS Code panel used to learn the list only from its own probe in the extension
        host — so a probe a corporate proxy swallowed left the user connected to a
        working endpoint with no way to switch model, the list missing rather than
        the capability.

        Sent after the greeting, not inside it: asking the endpoint is a network
        round trip, and the greeting is what takes the webview out of "connecting".
        Off-thread for the same reason — a slow endpoint must not hold the event loop
        that serves every other message. An endpoint that cannot enumerate itself
        sends nothing; the client keeps naming the active model.
        """
        try:
            models = await asyncio.get_event_loop().run_in_executor(
                None, self.worker.served_models)
        except Exception:
            return
        if not models:
            return
        try:
            await self.ws.send(json.dumps({"type": "served_models", "models": models}))
        except Exception:
            pass

    async def _handle_set_model(self, msg: dict) -> None:
        """Switch the served model mid-session.

        On success, report the new model plus the model-derived settings
        (thinking profile, enforcement) so the webview's controls stay coherent —
        the same state-not-narration pattern as ``/mode``. On failure, surface an
        error rather than silently keeping the old model.
        """
        model = (msg.get("model") or "").strip()
        if not model:
            await self.ws.send(json.dumps({
                "type": "error", "text": "  ✗ No model name given.\n",
            }))
            return
        error = next(iter(self.pool.set_model(model)), "")
        if error:
            await self.ws.send(json.dumps({"type": "error", "text": f"  ✗ {error}\n"}))
            return
        await self.ws.send(json.dumps({
            "type": "model_changed",
            "model": self.pool.model,
            "thinking": self.worker.get_thinking_profile(),
            "enforcement": self.worker.setting("enforcement"),
            "temperature": self.worker.get_temperature_state(),
        }))

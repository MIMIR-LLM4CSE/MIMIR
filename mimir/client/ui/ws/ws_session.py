"""The per-connection WebSocket session for the WS server.

``_Session`` owns one WebSocket connection: its chat history, the session store,
the drain loop that forwards worker events to the client, and the inbound-message
dispatch table. It talks to the background agent (``_AgentWorker`` in ``ws_worker``)
through that worker's thread-safe queues. Split out of ``ws_server.py``; see that
module's docstring for the wire protocol.
"""

from __future__ import annotations

# Import the shared runtime FIRST so its cwd bootstrap runs before config.constants
# (and the backend factory) capture the workspace root at import time.
from ._ws_runtime import (
    _MIMIR_DIR_WS,
    _todo_file_for_session,
    _write_active_session,
    context_budget_for,
    get_backend,
)
from ...query_engine.deferral import KIND_CALLS, take_deferred_calls
from ...query_engine.history import (
    carries_compaction_summary, reconcile_tool_pairs,
)
from .ws_worker import _AgentWorker
from .turn_commit import commit_answer, turn_messages
from .transcript_log import read_since
from .job_scan import (
    establish_baseline,
    has_baseline,
    mark_reported,
    scan_all_sessions,
)
from .detach import detach as detach_process
from . import server_registry
from .ws_pool import _AgentPool
from ...tool_execution import run_channel
from ...config import (
    TEMPERATURE_MAX, TEMPERATURE_MIN, THINKING_DEPTH_LABELS, parse_temperature,
    thinking_depth_from_label,
)

import asyncio
import json
import logging
import os
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation-only; avoids the runtime import cycle the methods dodge
    from .session_store import SessionMeta

# The websocket connection handle passed in by the server (websockets library type
# varies by version); kept as an untyped alias so the annotation resolves.
_WS = Any

logger = logging.getLogger(__name__)


# How much of a finished job's own summary a wake carries. Enough for an exit code
# and the tail of a build log; short of pasting a whole test suite into the history.
_WAKE_SUMMARY_LIMIT = 2000

# Per field, so one long value — a command built out of absolute paths, a log tail —
# cannot crowd the rest of the record out of the budget above.
_WAKE_VALUE_LIMIT = 500

# Fields the body leaves out: the head line of the wake already names the job and says
# how it ended, and repeating that as JSON is what made these messages unreadable.
_WAKE_SUMMARY_SKIP = frozenset({"job_key", "kind", "state"})

# Events that must reach the user no matter which conversation they belong to: each
# one is a question the agent is parked on, and filtering it as "foreign" (which it is,
# during a background-job wake in another session) would leave the turn waiting on an
# answer the user was never shown. They carry their ``session_id``, so the client can
# say which conversation is asking.
# ``prompt_expired`` rides with them: it takes one of these cards back off the screen,
# and filtering it as foreign would leave a card up for a wait that has ended.
_INTERACTION_EVENTS = frozenset({"approval", "user_question", "prompt_expired"})

# Events a reconnect may not throw away. Everything else a worker queues describes a
# turn — its tokens, its rows, its answer — and is debris once the socket that was
# drawing it is gone. These two describe a detached run instead: nothing re-emits them,
# their watcher is finished by the time they are read, and the conversation they belong
# to is idle precisely because it is waiting for them.
_DURABLE_EVENTS = frozenset({"job_complete", "job_checkin"})

# How many journal events one ``replay`` frame carries. The frame cap is 32 MiB, which a
# long detached run's transcript can exceed whole; chunking also lets the webview render
# the first of them while the rest are still arriving.
_REPLAY_CHUNK = 500

# The most a replay will resend. Past this the oldest are dropped and the frame says so:
# what a returning user needs is the end of the run, and an unbounded replay into the
# webview's reducer is how a twelve-hour absence wedges the panel.
_REPLAY_MAX_EVENTS = 5000


def _clip(text: str) -> str:
    """One field's value, cut in the middle so both of its ends survive.

    Which end carries the meaning depends on the field — a command says it at the
    front, a log tail at the back — and this layer does not know which it is holding.
    """
    if len(text) <= _WAKE_VALUE_LIMIT:
        return text
    head = _WAKE_VALUE_LIMIT * 2 // 3
    tail = _WAKE_VALUE_LIMIT - head
    return f"{text[:head]}… [cut: {len(text)} chars] …{text[-tail:]}"


def _render_field(key: str, value: object) -> str:
    """One recorded field as ``key: value``, or as an indented block when it has lines."""
    if isinstance(value, (dict, list)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)
    text = _clip(text).strip("\n")
    if "\n" in text:
        body = "\n".join(f"    {line}" for line in text.splitlines())
        return f"  {key}:\n{body}"
    return f"  {key}: {text}"


def _compact_summary(payload: dict) -> str:
    """A job's recorded result, as one field per line, cut to a budget.

    Passed through rather than interpreted: the client does not know what kind of job
    ran, so it hands the model what the server recorded, under the server's own field
    names. What it chooses is the shape — lines instead of a single JSON string, each
    value clipped on its own — because a wake is read by a person as well as a model,
    and one 800-character command should not be the whole of what either sees.
    """
    lines: list[str] = []
    budget = _WAKE_SUMMARY_LIMIT
    dropped = 0
    for key, value in payload.items():
        if key in _WAKE_SUMMARY_SKIP or value is None or value == "":
            continue
        line = _render_field(key, value)
        if lines and len(line) + 1 > budget:
            dropped += 1
            continue
        budget -= len(line) + 1
        lines.append(line)
    if dropped:
        lines.append(f"  [{dropped} more field(s) not shown]")
    if lines:
        return "\n".join(lines)
    # Everything it recorded was something the head line already said.
    try:
        return _clip(json.dumps(payload, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return _clip(str(payload))


def _reconcile(messages: list[dict]) -> list[dict]:
    """Repair assistant/tool pairing in a restored history.

    A resumed transcript can start on an orphaned ``role:"tool"`` (its assistant turn was
    trimmed away before the save) or end on a call whose result never landed, and strict
    tokenizers reject both. Best-effort: an unusable history is worse than an unrepaired
    one, so a failure here returns the messages untouched.
    """
    try:
        from ...query_engine.history import reconcile_tool_pairs
        return reconcile_tool_pairs(messages)
    except Exception:
        return messages


class _Session:
    """One WebSocket connection — owns a chat history and talks to the worker."""

    def __init__(self, ws: "_WS", pool: "_AgentPool") -> None:
        self.ws = ws
        self.pool = pool
        self.history: list[dict] = []  # LLM history — the working window, trimmed to fit
        # The same conversation, never trimmed. `history` is cut down whenever the budget
        # demands it, so saving that alone would lose whatever the window dropped; this is
        # what a resume reloads, so a session continues with everything it ever said.
        self.history_full: list[dict] = []

        try:
            from .session_store import SessionStore
        except ImportError:
            from mimir.client.ui.ws.session_store import SessionStore

        self.store = SessionStore()
        self._active_session_id: str | None = None
        self._display_messages: list[dict] = []  # serialisable UI messages
        # Metadata for a new session not yet saved to disk (no messages yet).
        self._unsaved_session_meta: "SessionMeta | None" = None
        # In-flight session-summary refresh (one at a time per connection).
        self._summary_task: asyncio.Task | None = None
        # Last `used_tokens` pushed to the client, so the periodic mid-turn refresh
        # only sends a frame when the number actually moved.
        self._last_context_usage: int | None = None
        # The context mode of the session on screen, as saved with it. What sizes the
        # window until this conversation has an agent to ask — a resumed session has no
        # agent until its first query, and budgeting it as compact sized a full-mode
        # window at 32k, which read as an overflow and trimmed history to match.
        self._resumed_context_mode: str = "full"
        # This connection's view of the stream, opened in run(). The pump owns the
        # draining; this is only where our share of it arrives.
        self._sub = None
        # Events withheld because they belong to another conversation, per session.
        # The one quiet-chat cause that is otherwise invisible — see _is_foreign_event.
        self._foreign_withheld: dict[str, int] = {}
        # How far into the journal the client's rendered transcript goes, as the client
        # last said. The watermark a re-attach replays from.
        self._rendered_seq = 0
        # Length of `history` at the moment the running turn was submitted, so the
        # answer can tell the turn's own messages from the prefix it inherited.
        self._submitted_len = 0
        # Wake turns running against sessions that are NOT on screen, as
        # {session_id: length of that session's history when it was submitted}. Keyed
        # rather than single: two jobs can finish into two different conversations
        # before either answers, and the second must not evict the first's bookkeeping.
        self._detached_turns: dict[str, int] = {}
        # Finished background jobs whose result has not yet been handed to a turn, as
        # {session_id: [job_complete, ...]}. A job that lands while a turn of its own
        # session runs is steered into that turn instead of queueing one of its own —
        # but a steer deposited past the turn's last step boundary is purged unread, so
        # an entry only leaves this map on *proof* of injection (or when a catch-up turn
        # carries it). Without that, three jobs finishing together produced three turns
        # and three near-identical final answers, one of them contradicting the last on
        # how many jobs there even were.
        self._pending_wakes: dict[str, list[dict]] = {}
        # The one check-in a conversation is holding back, as {session_id: job_checkin}.
        # A check-in carries "nothing to report" and nothing else, so unlike a wake it
        # never interrupts: while the owner is working or parked on a card, it waits
        # here and the *next* check-in overwrites it. One slot, not a list — a bulletin
        # is only worth reading while it is current, and a long turn that swallowed all
        # three should end with one fresh status line, not a backlog of stale ones.
        self._held_checkin: dict[str, dict] = {}
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
        # The active session's deferred interaction (query_engine.deferral): the card
        # to show and what answering it resumes. Mirrors the session file.
        self._pending_interaction: dict | None = None
        # Ids of deferred cards the user moved past without answering. Their card may
        # still be on screen; an answer to one must not reach the next live prompt.
        self._stale_prompt_ids: set[str] = set()
        # Sessions whose deletion was refused once because work of theirs was still
        # running. Asking again goes through: the warning is there to be read, not to
        # make the conversation undeletable, and the second click is the confirmation.
        self._delete_refused: set[str] = set()

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

        # Send ready immediately so the webview transitions out of "connecting".
        greeting = self._greeting()
        try:
            await self.ws.send(json.dumps(greeting))
        except Exception:
            self._sub.close()
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
            if self._summary_task is not None:
                self._summary_task.cancel()
            for task in (drain_task, jobs_task):
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
                        self.store.delete_session(meta.id)
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
        # Prepend the current unsaved session so the panel shows it immediately.
        if self._unsaved_session_meta is not None:
            existing_ids = {s.id for s in sessions}
            if self._unsaved_session_meta.id not in existing_ids:
                sessions = [self._unsaved_session_meta] + sessions
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
        """
        # Guarded: this is read for every row of every listing, and the send around it
        # swallows exceptions — so a hiccup here would blank the whole panel rather than
        # lose one dot.
        try:
            return {
                "running": self.pool.is_busy(session_id),
                "parked": self.pool.is_parked(session_id),
                "queued": self.pool.queued_position(session_id) is not None,
            }
        except Exception:
            return {}

    async def _create_new_session(self) -> None:
        session = self.store.new_session()
        # Don't save to disk yet — wait until there are messages to avoid
        # accumulating empty "New session" entries on every reconnect.
        self._active_session_id = session.id
        self.pool.set_active(session.id)
        _write_active_session(session.id)
        self.history = []
        self.history_full = []
        # Nothing to budget yet, and the agent built for it comes up in this mode.
        self._resumed_context_mode = "full"
        self._submitted_len = 0
        self._display_messages = []
        self._pending_interaction = None
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
            from .session_store import SessionMeta as _SM
        except ImportError:
            from mimir.client.ui.ws.session_store import SessionMeta as _SM
        self._unsaved_session_meta = _SM(
            id=session.id,
            title="",
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        try:
            await self.ws.send(json.dumps({
                "type": "session_loaded",
                "session_id": session.id,
                "title": session.title,
                "display_messages": [],
                "todos": [],
            }))
        except Exception:
            pass
        self._last_context_usage = None  # client cleared its bar on session_loaded
        await self._emit_context_usage()

    async def _load_session(self, session_id: str) -> None:
        session = self.store.load_session(session_id)
        # Whether this conversation already has an agent decides what gets restored
        # below: an agent that has been working holds the live carry context, and reloading
        # the saved snapshot over it throws away everything it learned since. Read before
        # the pointer moves, while `worker` still resolves the one being left.
        resuming_live_agent = self.pool.get(session.id) is not None
        self._active_session_id = session.id
        self._unsaved_session_meta = None  # switching to a persisted session
        self.pool.set_active(session.id)
        # Nothing is reset on a switch. Each conversation's approved paths are its own
        # file, so there is nothing of another conversation's to drop — and dropping
        # this one's would revoke grants its own turn may still be writing under.
        _write_active_session(session.id)
        # The untrimmed record is always the archive, whatever the model resumes on.
        # Sessions saved before this field existed only have the trimmed window — that
        # is still the best they have.
        self.history_full = list(session.llm_history_full or session.llm_history)
        # What the model actually resumes on. A window that already carries a
        # compaction summary *is* the conversation: the handoff note stands in for
        # everything that was cut, at a fraction of the tokens. Reloading the untrimmed
        # record over it threw that work away and handed the model a context that was
        # full before the user had typed anything — a long session came back at its own
        # pre-compaction size, and the first turn had to trim it all over again.
        #
        # Absent a summary the window is only the tail a front-trim left behind, so the
        # untrimmed record stays the faithful resume: its prefix was never summarized
        # anywhere, and dropping it would lose it outright.
        resumed = (session.llm_history
                   if carries_compaction_summary(session.llm_history)
                   else self.history_full)
        # Copy each message, not just the list: the budgeting pass mutates messages
        # in place (force-fit rewrites `content`, and the argument digest rewrites
        # `tool_calls`). Sharing the dicts made those writes reach straight into the
        # untrimmed record this line exists to preserve — the list-level protection
        # in the answer handler below cannot see through a shared reference.
        self.history = _reconcile([dict(m) for m in resumed])
        # Before the budget is read anywhere below: it sizes the window from this until
        # this conversation has an agent of its own to ask.
        self._resumed_context_mode = session.context_mode or "full"
        self._submitted_len = len(self.history)
        self._display_messages = list(session.display_messages)
        self._pending_interaction = session.pending_interaction
        self._rendered_seq = session.rendered_seq
        self._publish_title(session.id, session.title)
        if not resuming_live_agent:
            # Only when this conversation has no agent yet — the snapshot is how a
            # rebuilt agent picks up where the last one left off. An agent that is
            # already live has moved past it.
            self.worker.load_agent_state({"carry_context": session.carry_context})

        # If session had todos, restore them to disk so agent picks them up —
        # unless what is already on disk is newer than this snapshot (see below).
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
            # Only offer to resume if there are incomplete tasks remaining.
            pending_todos = [t for t in session.todos if not t.get("done")]
            try:
                await self.ws.send(json.dumps({
                    "type": "session_loaded",
                    "session_id": session.id,
                    "title": session.title,
                    "display_messages": session.display_messages,
                    "todos": [] if pending_todos else session.todos,
                    # A turn of this conversation that outlived the last connection is
                    # still producing. Said here rather than left to be inferred: the
                    # client shows the run as live, and — the part that is not
                    # cosmetic — knows a turn is open, so the answer that ends it is
                    # what hands the finished transcript back for saving.
                    "turn_running": self._running_turn_is_ours(),
                }))
            except Exception:
                pass
            if pending_todos:
                try:
                    await self.ws.send(json.dumps({
                        "type": "todo_prompt",
                        "items": session.todos,
                    }))
                except Exception:
                    pass
        else:
            try:
                await self.ws.send(json.dumps({
                    "type": "session_loaded",
                    "session_id": session.id,
                    "title": session.title,
                    "display_messages": session.display_messages,
                    "todos": session.todos,
                    "turn_running": self._running_turn_is_ours(),
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

    def _autosave_session(self, display_messages: list[dict]) -> None:
        """Persist current session state after an answer is delivered."""
        if self._active_session_id is None:
            return
        try:
            from datetime import datetime, timezone
            try:
                from .session_store import FullSession
            except ImportError:
                from mimir.client.ui.ws.session_store import FullSession

            if self.store.session_exists(self._active_session_id):
                session = self.store.load_session(self._active_session_id)
            else:
                # New session not yet on disk — preserve the original created_at.
                created_at = (
                    self._unsaved_session_meta.created_at
                    if self._unsaved_session_meta is not None
                    else datetime.now(timezone.utc).isoformat()
                )
                session = FullSession(
                    id=self._active_session_id, title="", created_at=created_at, updated_at=created_at
                )

            # Auto-title from first user message.
            if not session.title:
                for msg in display_messages:
                    if msg.get("role") == "user":
                        raw = msg.get("text") or msg.get("content", "")
                        session.title = raw[:60].strip()
                        break

            session.preview = session.title[:80]
            agent_state = self.worker.export_agent_state()
            session.llm_history = list(self.history)
            session.llm_history_full = list(self.history_full)
            session.display_messages = list(display_messages)
            session.carry_context = agent_state.get("carry_context", {})
            session.todos = self.worker._load_todos()
            session.pending_interaction = getattr(self, "_pending_interaction", None)
            # So a resume sizes the window the way this session was actually running,
            # rather than falling back to a default that may not be its own.
            session.context_mode = self.worker.get_context_mode(self._resumed_context_mode)
            # Never backwards: a replay is sized from this, and lowering it would resend
            # what the client already has.
            session.rendered_seq = max(session.rendered_seq, self._rendered_seq)
            # Persist deps sidecar alongside todos.
            try:
                import json as _json
                deps_file = os.path.join(
                    _MIMIR_DIR_WS, "sessions", self._active_session_id, "todo_deps.json"
                )
                if os.path.exists(deps_file):
                    with open(deps_file, "r", encoding="utf-8") as _f:
                        session.todo_deps = _json.load(_f)
                else:
                    session.todo_deps = []
            except Exception:
                session.todo_deps = []
            session.updated_at = datetime.now(timezone.utc).isoformat()
            self.store.save_session(session)
            self._unsaved_session_meta = None  # now persisted
        except Exception:
            pass  # Never crash the WS loop due to save failures.

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
            if not self.store.session_exists(session_id):
                return
            session = self.store.load_session(session_id)
            messages = list(session.display_messages)
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
            # Reload before writing: the turn may have saved again meanwhile.
            fresh = self.store.load_session(session_id)
            fresh.summary = summary
            fresh.summary_msgs = text_msgs
            fresh.summary_version = SUMMARY_VERSION if generated else PROVISIONAL_VERSION
            self.store.save_session(fresh)
            await self._send_sessions_list()
        except Exception:
            pass  # Descriptions are cosmetic — never disturb the session.

    # ── Context-budget helpers ────────────────────────────────────────────────

    def _ctx_budget(self) -> tuple:
        """Return (total_tokens, reserved_tokens) for the current context mode.

        For vLLM the window tracks the server's reported max_model_len (primed at
        startup so this runs against the cache, never blocking the event loop).
        """
        mode = self.worker.get_context_mode(self._resumed_context_mode)
        total, reserved, _, _ = context_budget_for(self.worker.model, mode)
        return total, reserved

    def _turn_messages(self, full: list[dict], submitted: int | None = None,
                       start: Any = None) -> list[dict]:
        """The slice of *full* this turn produced — what the archive has yet to record.

        The boundary comes from the loop, which is the only place it is knowable. The
        length we submitted is not an index into *full* once the in-turn budget pass has
        rewritten the list: it evicts old tool results, replaces the middle with a summary
        and then repairs the assistant↔tool pairing those break. Every such rewrite shifts
        the prefix, and a stale boundary re-archives whatever it shifted past or drops
        whatever it shifted over — cutting through an assistant↔tool pair on the way, which
        is how a record ends up holding tool results with no call in front of them.

        Falls back to the submitted length, and then to the answer alone, when the loop
        cannot place the boundary — a turn long enough to have its own opening message
        summarized away. Better a turn recorded by its answer than a record quietly
        interleaved with a copy of an older one.
        """
        if start is None:
            hook = getattr(self.worker, "last_turn_start", None)
            start = hook() if hook else None
        fallback = self._submitted_len if submitted is None else submitted
        return turn_messages(full, fallback, start=start)

    async def _emit_context_usage(self) -> None:
        """Push a context_usage event to the WS client (best-effort, never raises)."""
        try:
            total, reserved = self._ctx_budget()
            # While a query runs, the transcript that matters is the agent's in-flight
            # one: `self.history` only gains the turn once the answer lands, so the bar
            # would sit frozen for the whole run.
            messages = self.worker.live_history()
            if messages is None:
                messages = self.history
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
                "history_messages": len(self.history),
                "history_messages_full": len(self.history_full),
            }))
        except Exception:
            pass

    # ── Drain loop ────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        """Forward this connection's share of the stream to the WS client.

        ``run()`` subscribes before the handshake so nothing can be missed; this is the
        safety net for a caller that starts the loop on its own.
        """
        if self._sub is None:
            self._sub = self.pool.bus.subscribe()
        _todo_mtime: float = 0.0  # last known mtime of the session's todo file

        async def _check_and_push_todos() -> None:
            nonlocal _todo_mtime
            try:
                todo_path = _todo_file_for_session(self._active_session_id)
                mtime = os.path.getmtime(todo_path) if os.path.exists(todo_path) else 0.0
                if mtime != _todo_mtime:
                    _todo_mtime = mtime
                    items = self.worker._load_todos()
                    if items:
                        await self.ws.send(json.dumps({"type": "todo", "items": items}))
            except Exception:
                pass

        _last_ctx_tick = 0.0

        async def _tick_context_usage() -> None:
            """Refresh the bar during a running turn, at most once a second.

            A turn can run dozens of tool calls over several minutes; without this the
            bar only moves when the answer lands and looks frozen for the whole run.
            """
            nonlocal _last_ctx_tick
            if not self._running_turn_is_ours():
                return  # a detached wake turn's usage is not this session's
            now = time.monotonic()
            if now - _last_ctx_tick < 1.0:
                return
            _last_ctx_tick = now
            await self._emit_context_usage()

        _last_progress_tick = 0.0

        async def _tick_run_channels() -> None:
            """Push what each blocking run is doing, at most once a second.

            A run that blocks the turn for twenty minutes shows a spinner and nothing
            else; the server publishes its phase on the run channel precisely because
            the tool call it belongs to cannot answer until it is over. Transient, so
            deliberately not appended to the transcript — like the context bar, it
            describes a moment rather than recording one.
            """
            nonlocal _last_progress_tick
            rows = self._rows_for(self._active_session_id)
            if not rows:
                return
            now = time.monotonic()
            if now - _last_progress_tick < 1.0:
                return
            _last_progress_tick = now
            # Only the conversation on screen: a bar describes a moment the user is
            # looking at, and a turn running elsewhere has nowhere to draw one. Its
            # progress is on its own run channel either way, and is read when the user
            # switches to it.
            #
            # A name serving two running rows cannot be attributed to either of them.
            # Both publishers are non_batch, so this is insurance, not a live case.
            sent = self._sent_progress.setdefault(self._active_session_id or "", {})
            names = list(rows.values())
            for call_id, name in list(rows.items()):
                if names.count(name) > 1:
                    continue
                try:
                    run = run_channel.current_run(name, self._active_session_id)
                except Exception:
                    continue
                if not run:
                    continue
                phase = str(run.get("phase") or "")
                percent = run.get("percent")
                if not isinstance(percent, (int, float)):
                    percent = None
                # Nothing-to-say is skipped only while nothing was said: once a bar
                # is up, its retraction (the build ended, the command went on) is a
                # change the row must hear about, or the bar stays frozen.
                if sent.get(call_id, ("", None)) == (phase, percent):
                    continue
                sent[call_id] = (phase, percent)
                try:
                    await self.ws.send(json.dumps({
                        "type": "tool_progress", "id": call_id,
                        "phase": phase, "percent": percent,
                    }))
                except Exception:
                    return

        while True:
            # Read from this connection's subscription, not from the workers: the pump
            # owns the draining, so that output is recorded whether or not this socket
            # exists. Per-conversation order comes free — each worker owns its own FIFO
            # — and the interleaving between them is harmless because every event
            # carries the session it was produced for.
            events: list[tuple[dict, dict]] = []
            while True:
                try:
                    events.append(self._sub.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            for ev, extras in events:
                # Which conversation this belongs to. Everything below addresses a
                # session rather than "the session", because more than one may be
                # producing at this instant.
                owner = ev.get("session_id") or self._active_session_id
                if ev.get("type") in _DURABLE_EVENTS:
                    # Routed by the session that launched the job rather than the one
                    # on screen, so it is handled ahead of the foreign-event filter.
                    checkin = ev.get("type") == "job_checkin"
                    foreign = self._is_foreign_event(ev)
                    # Decided once, before the send: the send yields, and a turn that
                    # ended meanwhile would otherwise have the client and the handler
                    # disagree about whether a new turn began.
                    steer = False if checkin else self._wake_steers(ev)
                    # Handled BEFORE the send, and this order is the point. The event
                    # has already left ``out_q`` and nothing re-emits it, so a socket
                    # that dies on the send — the ordinary end of a connection — must
                    # not be able to take the handler down with it: the agent behind
                    # the run is fine, and the run still has to wake it. The send is
                    # the notification; this is the work.
                    started = await self._handle_durable_event(ev, checkin=checkin,
                                                               steer=steer)
                    try:
                        # A wake starts a turn nobody pressed send for. The client
                        # marks itself busy on its own submit, so without this it has
                        # no way to know one began, and offers no way to stop it. It
                        # cannot work the answer out for itself either: the wake may
                        # resume a session that is not the one on screen, and marking
                        # the visible chat busy for a turn running elsewhere leaves a
                        # stop button that stops nothing.
                        # A wake steered into the turn already running starts nothing,
                        # and saying it did makes the client drop that turn's live
                        # rows — a build launched in the same step lost its row, and
                        # with it every progress update addressed to it.
                        await self.ws.send(json.dumps(
                            {**ev, "resumes_active_session": not foreign and started},
                            default=str))
                    except Exception:
                        return
                    continue
                if self._is_foreign_event(ev):
                    # Counted here rather than in the predicate, which must stay a
                    # predicate. This is the one way the chat can go quiet that leaves
                    # no trace: the event is withheld from the socket and journaled
                    # under its own session, so the conversation looks stalled while
                    # the record fills — and a reconnect replays it and it appears.
                    if owner:
                        self._foreign_withheld[owner] = (
                            self._foreign_withheld.get(owner, 0) + 1)
                    # A turn running in a conversation that is not on screen. Its output
                    # has nowhere to be drawn, but it still happened: it goes to that
                    # conversation's own log, so switching to it shows the whole turn and
                    # not only the answer that ended it. (Progress is excluded for the same
                    # reason it is below — it describes a moment.)
                    if ev.get("type") in ("answer", "error"):
                        if ev.get("type") == "answer":
                            await self._persist_detached_answer(ev, extras)
                        await self._admit_waiting()
                    continue
                # Unwrap embedded JSON events (e.g. diff) from output lines.
                if ev.get("type") == "output":
                    text = ev.get("text", "").strip()
                    if text.startswith("{") and text.endswith("}"):
                        try:
                            inner = json.loads(text)
                            if isinstance(inner, dict) and "type" in inner:
                                ev = inner
                        except json.JSONDecodeError:
                            pass
                # Logged before the send: an event the client never received still
                # happened, and the log is the record of the run, not of the socket.
                # Progress is the exception: a watcher ticks for the whole life of a
                try:
                    await self.ws.send(json.dumps(ev, default=str))
                except Exception:
                    return
                if ev.get("type") == "tool_call" and ev.get("divertible"):
                    # Only divertible rows: they are exactly the ones a channel can
                    # answer for, and the map is what both the divert click and the
                    # progress poller resolve through.
                    self._rows_for(owner)[str(ev.get("id") or "")] = str(ev.get("name") or "")
                elif ev.get("type") == "tool_result":
                    self._forget_row(str(ev.get("id") or ""), owner)
                if ev.get("type") == "error":
                    # A turn failed (often a context-overflow 400) and no `answer`
                    # event follows, so refresh the context bar here or it keeps
                    # showing pre-failure usage and never reflects the overflow.
                    self._rows_for(owner).clear()
                    self._sent_progress.setdefault(owner or "", {}).clear()
                    await self._emit_context_usage()
                    await self._admit_waiting()
                if ev.get("type") == "file_progress":
                    # Push accumulated batch_status for any files already written
                    # in this turn so the BatchReviewBar appears/updates mid-turn.
                    try:
                        files = self.worker._build_batch_status()
                        if files:
                            await self.ws.send(json.dumps({"type": "batch_status", "files": files}))
                    except Exception:
                        pass
                    # Pause so the browser can render the "writing..." card before
                    # the write completes and the diff / answer messages arrive.
                    await asyncio.sleep(0.1)
                if ev.get("type") == "diff":
                    # A file write just completed. Push an updated batch_status
                    # immediately so the BatchReviewBar appears mid-turn rather
                    # than waiting for the agent to finish.
                    try:
                        files = self.worker._build_batch_status()
                        if files:
                            await self.ws.send(json.dumps({"type": "batch_status", "files": files}))
                    except Exception:
                        pass
                if ev.get("type") == "steer_injected":
                    # The loop confirming what it took in is the only proof a steered
                    # wake was actually read. Anything still pending after this is
                    # carried by the catch-up turn below.
                    self._drop_injected_wakes(str(ev.get("text") or ""))
                if ev.get("type") == "answer":
                    # The turn is over: no row of it is still running, so nothing is
                    # left for a channel to report on. Its own conversation's rows —
                    # another turn may be running in another one, and its rows are live.
                    self._rows_for(owner).clear()
                    self._sent_progress.setdefault(owner or "", {}).clear()
                    # In full-context mode keep the structured transcript (tool_calls +
                    # results + answer, chain-of-thought stripped) so the model recalls
                    # the tools it ran, matching the CLI chat loop. Falls back to the
                    # flattened answer otherwise.
                    full, start = self._answer_transcript(extras)
                    context_mode = getattr(self.worker._agent, "context_mode", "full")
                    # Left and come back before it answered: it lands here after all.
                    self._detached_turns.pop(self._active_session_id, None)
                    if extras.get("_deferred"):
                        self._pending_interaction = extras["_deferred"]
                    if full is not None and context_mode == "full":
                        # Keep only what the turn itself produced. The loop may have
                        # trimmed or compacted the prefix it inherited from us, and that
                        # prefix is exactly what the untrimmed record exists to hold —
                        # so it must not be overwritten by the shortened copy. A turn
                        # whose own messages were compacted away still has its answer,
                        # which is the part worth keeping.
                        added = self._turn_messages(full, start=start)
                        # Copied, for the same reason as the load path above: a later
                        # turn's budgeting rewrites `content` / `tool_calls` in place,
                        # and these dicts would otherwise be the archive's own.
                        self.history_full.extend(dict(m) for m in added)
                        self.history = full
                    else:
                        answer_msg = {"role": "assistant", "content": ev.get("text", "")}
                        self.history.append(answer_msg)
                        self.history_full.append(dict(answer_msg))
                    if ev.get("text"):
                        self._display_messages.append({
                            "role": "agent",
                            "kind": "text",
                            "text": ev.get("text", ""),
                        })
                    self._autosave_session(list(self._display_messages))
                    # Set aside while this conversation was still on screen: the card
                    # it was parked on comes straight back.
                    if extras.get("_deferred"):
                        await self._resend_deferred_prompt()
                    # Once the turn has landed, so the description says what was *done*
                    # rather than what was asked, and the model call no longer competes
                    # with the query the user is waiting on.
                    self._schedule_summary_refresh()
                    await self._emit_context_usage()
                    # This conversation stopping work may let the pool release a different
                    # idle one to make room for whoever is waiting.
                    await self._admit_waiting()
                    # Sent directly rather than via out_q, so the snapshot dict is read
                    # *after* any batch_review_accept that arrived mid-run cleared it.
                    try:
                        files = self.worker._build_batch_status()
                        await self.ws.send(json.dumps({"type": "batch_status", "files": files}))
                    except Exception:
                        pass
                    # Last, so the catch-up turn is handed the history this answer just
                    # wrote rather than the one it inherited. Jobs that finished during
                    # the turn and were never taken in leave together, as one turn.
                    await self._flush_pending_wakes(self._active_session_id)
                    # And the bulletin this conversation held back while it was
                    # working, now that the answer the user waited for has landed.
                    # After the wakes, which supersede it.
                    await self._flush_held_checkin(self._active_session_id)
            await asyncio.sleep(0.005)
            await _check_and_push_todos()
            await _tick_context_usage()
            await _tick_run_channels()

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
        "resume_plan": "_handle_resume_plan",
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
    }

    async def _handle(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return
        handler_name = self._MSG_HANDLERS.get(msg.get("type"))
        if handler_name is None:
            return  # unknown/missing type — ignore, as the old fall-through did
        await getattr(self, handler_name)(msg)

    @staticmethod
    def _wake_text(ev: dict) -> str:
        """Build the auto-resume instruction from a job_complete event.

        The client does not know what a job *does*. A detached run can be a two-hour
        compile, a Slurm batch or a proxy optimization, and the only thing this layer
        holds about it is the descriptor the server handed over. So the wake states
        the fact and passes the payload through: what the model should do next comes
        from the job's own recorded result, never from an instruction invented here.
        Naming another server's ops in this function is how a build once got told to
        review proxy results and continue an optimization loop that did not exist.

        The one tool name it may use is ``status_op``'s, and only because that is
        registry data travelling on the descriptor — the same reason the watcher can
        poll generically. It is the last resort, for a job that recorded no summary.
        """
        job_key = ev.get("job_key", "?")
        state   = ev.get("state", "done")
        kind    = ev.get("kind")
        summary = ev.get("summary") if isinstance(ev.get("summary"), dict) else {}

        what = f"Background job '{job_key}'" + (f" ({kind})" if kind else "")
        if state == "crashed":
            head = f"{what} crashed."
        elif state == "unknown":
            why = ev.get("reason") or "its status stopped being readable"
            head = f"{what} can no longer be tracked: {why}."
        else:
            head = f"{what} finished."

        # Payload conventions, shown only where the server put them — keys, not tool
        # names, so a job that carries none is described by its summary alone.
        marks = []
        if summary.get("verdict"):
            marks.append(f"verdict={summary['verdict']}")
        best = summary.get("best") or {}
        if isinstance(best, dict) and best.get("primary_value") is not None:
            marks.append(f"best {summary.get('primary_metric', 'primary')}="
                         f"{best.get('primary_value')}")
        if marks:
            head = f"{head} {' '.join(marks)}"

        # A next step the *server* wrote is an instruction from something that knows
        # the job; relayed verbatim.
        next_step = summary.get("next_step")
        if next_step:
            return f"{head} {next_step}"
        if summary:
            return (f"{head} Here is what it recorded — read it, then carry on with "
                    f"the work it was part of:\n{_compact_summary(summary)}")
        status_tool = (ev.get("status_op") or {}).get("tool")
        if status_tool:
            return (f"{head} It recorded no result of its own; read its state with "
                    f"'{status_tool}', then carry on with the work it was part of.")
        return f"{head} Carry on with the work it was part of."

    def _wake_owner(self, ev: dict) -> str | None:
        """The session a finished job belongs to: the one that launched it."""
        return ev.get("session_id") or self._active_session_id

    def _wake_steers(self, ev: dict) -> bool:
        """True when *ev* will be handed to a turn already running, not start one.

        Asked of the owner's own agent: a wake belongs to its conversation whether or not
        anyone is reading it, and several conversations can be working at once.
        ``_running_turn_is_ours`` answers a different question — is the turn in flight the
        one on screen — which is the right test for a message the user typed and the wrong
        one here.
        """
        owner = self._wake_owner(ev)
        return bool(owner) and self.pool.is_busy(owner)

    @staticmethod
    def _checkin_text(ev: dict) -> str:
        """Build the check-in instruction from a job_checkin event.

        Under the same rule as :meth:`_wake_text` — it names no tool of any server and
        interprets nothing, passing on whatever the status op chose to report. What it
        adds is a ceiling on the answer. A check-in exists so a run that went wrong in
        its third minute is not discovered in its hundred-and-twentieth; it is not an
        occasion to restate the plan, re-answer the question the job was launched for,
        or go and look at something. Said plainly here because the model has no other
        way to tell this turn apart from a completion wake, which wants the opposite.

        Reading the run is still open to it where the status says something is off: the
        dispatch guard refuses the *status* op of a watched job (this event already
        carries that answer) and leaves the summary op alone.
        """
        jobs = [j for j in (ev.get("jobs") or []) if isinstance(j, dict)]
        lines = []
        for job in jobs:
            key = job.get("job_key", "?")
            kind = f" ({job['kind']})" if job.get("kind") else ""
            state = job.get("state") or "running"
            bits = [state]
            if job.get("phase"):
                bits.append(f"phase={job['phase']}")
            if isinstance(job.get("percent"), (int, float)):
                bits.append(f"{job['percent']}%")
            lines.append(f"- '{key}'{kind}: {', '.join(bits)}")
        body = "\n".join(lines) or "- (no status recorded yet)"
        return (
            "Background check-in. Still running in this conversation:\n"
            f"{body}\n"
            "If this looks healthy, say so in ONE short line and stop — no tool call, "
            "no restating the plan, and do not re-answer the question these runs were "
            "launched for. If it does not — no progress since the last check, a status "
            "that stopped being readable, a phase that should have moved on by now — "
            "say what is wrong and what you are doing about it."
        )

    async def _handle_durable_event(self, ev: dict, *, checkin: bool,
                                    steer: bool) -> bool:
        """Route a job event to its conversation. True when it started a turn there.

        The drain loop needs that answer before it sends: the client marks itself busy
        on its own submit, so a turn nobody pressed send for is one it can only learn
        about from ``resumes_active_session``.
        """
        if checkin:
            return await self._handle_job_checkin(ev)
        return await self._handle_job_complete(ev, steer=steer)

    async def _handle_job_checkin(self, ev: dict) -> bool:
        """Report on runs still going, without ever interrupting. True if a turn began.

        A completion wake carries a result the running turn needs; a check-in carries
        "nothing to report". Steering that into a turn the user is waiting on makes the
        agent answer about the job instead of the question it was asked — and the
        silence a check-in exists to break is not there in the first place while the
        user is watching it work. So an owner that is busy, or parked on a card a person
        has to answer, gets nothing now: the bulletin waits in ``_held_checkin`` and the
        next one overwrites it, so a long turn ends with one current status line rather
        than a backlog of stale ones. :meth:`_flush_held_checkin` delivers it once the
        turn lands.
        """
        owner = self._wake_owner(ev)
        if not owner:
            return False
        if self.pool.is_busy(owner) or self.pool.is_parked(owner):
            self._held_checkin[owner] = ev
            return False
        return await self._deliver_checkin(owner, ev)

    async def _deliver_checkin(self, owner: str, ev: dict) -> bool:
        """Start the short turn a check-in asks for. True when one was submitted."""
        text = self._checkin_text(ev)
        worker = self.pool.get(owner)
        if worker is None:
            # Nothing to wake. Unlike a completion wake there is nothing to preserve
            # either: the run is still going and its own wake is still to come.
            logger.warning("check-in for session %s dropped: it has no agent", owner)
            return False
        self._record_wake(owner, text, [ev], entry_type="job_checkin")
        if owner != self._active_session_id:
            try:
                session = self.store.load_session(owner)
            except Exception:
                return False
            return await self._submit_detached(session, text)
        self.history.append({"role": "user", "content": text})
        self.history_full.append({"role": "user", "content": text})
        self._autosave_session(list(self._display_messages))
        self._submitted_len = len(self.history)
        worker.submit_query(text, list(self.history), session_id=owner)
        return True

    async def _flush_held_checkin(self, owner: str | None) -> None:
        """Deliver the bulletin a conversation held back, if it is still worth reading.

        Called where a turn of *owner* has just landed. Dropped rather than delivered
        when the runs it described have since finished: their completion wakes say
        everything this would have, and better — and one of them is about to be flushed
        into the very next turn.
        """
        ev = self._held_checkin.pop(owner, None)
        if ev is None or self._pending_wakes.get(owner):
            return
        worker = self.pool.get(owner)
        if worker is None or not worker.watched_job_keys():
            return
        await self._deliver_checkin(owner, ev)

    async def _handle_job_complete(self, ev: dict, steer: bool | None = None) -> bool:
        """Hand a finished background job to a turn — the running one where possible.

        The event was already forwarded to the client (notification) by the drain loop.

        The news goes to the session that *launched* the job, which the watcher
        recorded. A two-hour build outlives the conversation on screen, and dropping
        its result into whatever the user happens to be reading puts an answer in a
        conversation that never asked the question.

        Where it goes *within* that session depends on whether it already has a turn in
        flight. If it does, the job is steered into it: the turn learns the run finished
        at its next step boundary and carries on, costing no extra turn and no second
        final answer. If it does not, a turn is started, carrying every job still
        waiting — so a burst that finished during the last turn arrives as one turn
        rather than one apiece.

        Returns whether a turn actually started here, which is what the drain loop puts
        on the wire as ``resumes_active_session``: a steer starts none, and neither does
        a flush that had to keep its wakes. Answering "yes" for either leaves the chat
        marked busy, with a stop button, for a turn that is not running.
        """
        owner = self._wake_owner(ev)
        if steer is None:
            steer = self._wake_steers(ev)
        # Each entry remembers whether the user has already been shown this job, so a
        # later flush re-tells the *model* (a steer may never have been read) without
        # writing the notice and the log line a second time.
        item = {"ev": ev, "told": False}
        self._pending_wakes.setdefault(owner, []).append(item)
        if steer:
            wake = self._wake_text(ev)
            self._record_wake(owner, wake, [ev])
            item["told"] = True
            # Left pending deliberately: a steer is only known to have arrived when the
            # loop says so, and until then this job still needs a turn of its own.
            #
            # Steered into the agent of the conversation that launched the job, which is
            # not necessarily the one on screen: aiming anywhere else injects a job's
            # result into whatever conversation the user happens to be reading.
            owner_worker = self.pool.get(owner)
            if owner_worker is not None:
                owner_worker.submit_steer(wake)
            return False
        return await self._flush_pending_wakes(owner)

    def _record_wake(self, owner: str | None, wake: str, events: list[dict],
                     entry_type: str = "job_wake") -> None:
        """Show a wake and log it — once per job, whichever route carried it.

        Deliberately does NOT touch the history: which message a turn is *given* is the
        flush's business, and a job handed to a running turn arrives in that turn's own
        messages instead. Writing here as well is what put the same wake in the history
        twice, once as a steer and once inside the combined catch-up message.

        Without the log line, a job reported through the running turn left no
        ``job_wake`` at all — and the log is what the mechanism is checked with, so it
        would have undercounted exactly the runs it handled best.

        Which conversation gets it is decided by *owner*, not by what is on screen: a
        detached session's turn can be running and absorb a second job of its own.
        """
        # The entry type separates a bulletin from a wake in the transcript: the count
        # of ``job_wake`` lines is how the mechanism is audited against the jobs that
        # recorded an exit code, and filing three check-ins under it per run would make
        # a mechanism that lost half its wakes look like one that delivered four times
        # too many.
        text_of = self._checkin_text if entry_type == "job_checkin" else self._wake_text
        entries = [{"type": entry_type, "text": text_of(e), "job": e.get("job_key")}
                   for e in events]
        # A bulletin leaves no bubble. The client is sent the ``job_checkin`` event
        # itself and renders the news that runs are still there; a stored copy of the
        # instruction that *asks* the model for its one line is addressed to the model,
        # not to a reader, and it was kept for ever — a conversation checked on twenty
        # times reopened on twenty blocks of it, every one above the answer it had
        # produced. The record below still gets it, which is what the mechanism is
        # audited from.
        note = None if entry_type == "job_checkin" else {
            "role": "system", "kind": "text", "text": f"🔔 {wake}"}
        if owner == self._active_session_id:
            for entry in entries:
                self.pool.bus.record_client_event(owner, entry)
            if note is not None:
                self._display_messages.append(note)
                self._autosave_session(list(self._display_messages))
            return
        try:
            session = self.store.load_session(owner)
        except Exception:
            return
        if note is not None:
            session.display_messages.append(note)
            try:
                self.store.save_session(session)
            except Exception:
                return
        for entry in entries:
            self.pool.bus.record_client_event(owner, entry)

    def _drop_injected_wakes(self, text: str) -> None:
        """Forget the pending jobs whose wake *text* the running turn just took in.

        Matched on the job key rather than the whole message because the loop reports
        what it injected, not which event it came from — and a wake names its job key,
        so the match is exact. Anything not matched here stays pending and is carried by
        the catch-up turn, which is what keeps an unread steer from losing a run.
        """
        for owner, items in list(self._pending_wakes.items()):
            kept = [i for i in items
                    if str(i["ev"].get("job_key") or "\x00") not in text]
            if kept:
                self._pending_wakes[owner] = kept
            else:
                self._pending_wakes.pop(owner, None)

    async def _flush_pending_wakes(self, owner: str | None) -> bool:
        """Start one turn for every job of *owner* still waiting to be reported.

        The turn is given every pending job, including ones already steered: a steer is
        only known to have been read when the loop says so, and re-telling a run is
        recoverable where losing one is not. Only the jobs the user has not already been
        shown are recorded again.

        The message is the concatenation of each job's own wake text. Nothing is
        summarised into a sentence of this layer's own: a wake may relay a next step the
        *server* wrote, and a précis would drop it — the same reason :meth:`_wake_text`
        passes the payload through rather than describing it.

        Returns whether a turn started, so a caller that has to tell the client can.
        """
        items = self._pending_wakes.pop(owner, [])
        if not items:
            return False
        events = [i["ev"] for i in items]
        fresh = [i["ev"] for i in items if not i["told"]]
        wake = "\n\n".join(self._wake_text(e) for e in events)
        if owner != self._active_session_id:
            if await self._resume_detached_session(owner, wake, fresh):
                return True
            self._keep_pending(owner, items)
            return False
        # The agent of *owner*, which here is the conversation on screen — but asked of
        # the pool rather than read off ``self.worker``, because a worker released while
        # this conversation sat idle would hand back the detached stand-in, whose queue
        # no loop reads. A wake put there is a wake that never happens.
        worker = self.pool.get(owner)
        if worker is None:
            logger.warning("wake for session %s held: it has no agent to run it", owner)
            self._keep_pending(owner, items)
            return False
        if fresh:
            self._record_wake(owner, "\n\n".join(self._wake_text(e) for e in fresh), fresh)
        self.history.append({"role": "user", "content": wake})
        self.history_full.append({"role": "user", "content": wake})
        self._autosave_session(list(self._display_messages))
        self._submitted_len = len(self.history)
        worker.submit_query(wake, list(self.history), session_id=owner)
        return True

    def _keep_pending(self, owner: str | None, items: list[dict]) -> None:
        """Put wakes back after a flush that could not deliver them.

        The flush takes them off the map before it can know whether it will succeed, and
        every way it fails — a session deleted under it, a store that will not write, an
        agent released out from under it — is a finished run reported to nobody, which is
        the one outcome this whole mechanism exists to prevent. Prepended: they were
        waiting before whatever arrived while this ran.

        ``told`` is preserved, so a retry re-tells the model without showing the user the
        same notice twice.
        """
        if not items:
            return
        self._pending_wakes[owner] = items + self._pending_wakes.get(owner, [])

    async def _resume_detached_session(self, session_id: str, wake: str,
                                       fresh: list[dict]) -> bool:
        """Run a job wake against a stored session that is not the one on screen.

        The conversation lives on disk, not in this object's ``history``, so the wake
        is appended there and the turn is submitted against that copy. Its events come
        back stamped with *this* session id and are filtered out of the socket's
        stream by :meth:`_is_foreign_event`; :meth:`_persist_detached_answer` is what
        writes the answer back.

        False when the turn did not start — a session deleted under it, a store that
        will not write, an agent that is gone. The caller puts the wakes back rather
        than letting the run go unreported.
        """
        keys = ", ".join(str(e.get("job_key", "?")) for e in fresh) or "?"
        try:
            session = self.store.load_session(session_id)
        except Exception:
            await self._notify(f"Background job '{keys}' finished, but "
                               f"the session that launched it is gone.")
            return False
        if fresh:
            shown = "\n\n".join(self._wake_text(e) for e in fresh)
            session.display_messages.append(
                {"role": "system", "kind": "text", "text": f"🔔 {shown}"})
        if not await self._submit_detached(session, wake):
            return False
        for e in fresh:
            self.pool.bus.record_client_event(
                session_id, {"type": "job_wake", "text": self._wake_text(e),
                             "job": e.get("job_key")})
        await self._notify(f"🔔 “{session.title or session_id}” resumed in the "
                           f"background: {wake.split(chr(10))[0]}")
        await self._send_sessions_list()
        return True

    async def _submit_detached(self, session, text: str) -> bool:
        """Append *text* to a stored conversation and run a turn of it. False if it could not.

        The agent is resolved through the pool by the session that *owns* the turn, not
        read off ``self.worker``: that property answers for the conversation on screen,
        which by definition is not this one. It would hand back another conversation's
        agent — running this turn behind that one's work, and leaving ``is_busy`` saying
        the wrong thing about both — or, where the visible conversation has not been
        asked anything yet, the detached stand-in, whose queue no loop reads.
        """
        session_id = session.id
        worker = self.pool.get(session_id)
        if worker is None:
            logger.warning("turn for session %s not started: it has no agent", session_id)
            return False
        msg = {"role": "user", "content": text}
        session.llm_history.append(msg)
        session.llm_history_full.append(dict(msg))
        try:
            self.store.save_session(session)
        except Exception:
            logger.warning("turn for session %s not started: its session would not save",
                           session_id, exc_info=True)
            return False
        # What the turn was handed, so the answer path can tell the turn's own messages
        # from the prefix it inherited — the same bookkeeping ``_submitted_len`` does
        # for the active session.
        self._detached_turns[session_id] = len(session.llm_history)
        worker.submit_query(text, list(session.llm_history), session_id=session_id)
        return True

    async def _report_ended_jobs(self) -> None:
        """Announce every detached run that finished while nothing was watching it.

        This is what makes coming back enough. A background run survives anything — its
        own process session, a trap that records the exit code, a descriptor on disk —
        but the *watcher* that promised to report it is a task on a worker's loop, and a
        restart takes it with them. Re-making that promise already existed, through a
        status tool returning ``background_jobs``, but it needed a turn in which somebody
        asked; here nobody has to.

        Handed to :meth:`_handle_job_complete`, deliberately: a run reported late must be
        indistinguishable from one reported on time, so it goes through the same routing,
        the same wake text and the same coalescing. A conversation with no agent yet keeps
        its wake pending rather than losing it, and the next flush delivers it.

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
        recent one, and unlike a missed wake unbounded.
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
                mark_reported(job)
                await self._handle_job_complete({
                    "type": "job_complete",
                    "job_key": job.job_key,
                    "state": job.state,
                    "session_id": session_id,
                    "summary": {"command": job.command, "exit_code": job.exit_code},
                    "status_op": job.status_op(),
                })

    async def _restore_detached_autonomy(self) -> None:
        """Come back to the conversation in the mode it was left running under.

        A reattach is a window opening onto a run that never stopped, so the run's own
        level is the truth and the panel has to show it. Two things can disagree with it
        otherwise: a webview arrives on its default, and a worker rebuilt during the
        absence comes up on the pool-wide record rather than the level chosen for this
        session — which would quietly drop a run from ``auto`` to ``manual`` and park it
        at its next sensitive call, with nothing said.

        Scoped to the run, not persisted past it: the level is read from the registry
        entry, which is cleared on a clean stop and liveness-checked. That is what keeps
        this from becoming the thing the approval mode is deliberately never persisted
        for — a mode that suppresses prompts being inherited by a later session that
        never asked for it.
        """
        entry = server_registry.read()
        if not entry or not entry.get("detached"):
            return
        level = str(entry.get("autonomy") or "manual")
        if level not in ("manual", "auto", "auto_all"):
            return
        # Recorded as well as applied, so a worker built after this comes up on it.
        self._apply_setting("set_approval_mode", level)
        try:
            await self.ws.send(json.dumps({"type": "approval_mode", "mode": level}))
            if level != "manual":
                await self._notify(
                    f"Still running under \u201c{level}\u201d, the level it was "
                    f"detached with.")
        except Exception:
            return
        logger.info("reattach: autonomy restored to %s from the detached run", level)

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
            self._sub.min_seq = max(self._sub.min_seq, self._rendered_seq)
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
        self._sub.min_seq = max(self._sub.min_seq, through)
        logger.info("replay: session %s resent %d event(s) from seq %d",
                    session_id, len(events), self._rendered_seq)

    async def _notify(self, text: str) -> None:
        """One line of chrome for the client, outside any conversation. Never raises."""
        try:
            await self.ws.send(json.dumps({"type": "output", "text": f"  {text}\n"}))
        except Exception:
            pass

    def _answer_transcript(self, extras: dict) -> tuple[list | None, Any]:
        """The finished turn's transcript and boundary, from its answer when carried."""
        if "_full" in extras:
            return extras["_full"], extras.get("_turn_start")
        return self.worker.full_history(), None

    async def _persist_detached_answer(self, ev: dict, extras: dict | None = None) -> None:
        """Write a detached turn's answer into its own session file.

        The counterpart of :meth:`_resume_detached_session`. Without it the turn would
        run, cost its tokens, and vanish: its ``answer`` event belongs to a session
        this socket is not showing, so the drain loop's normal answer handling — which
        writes to ``self.history`` — must not touch it.
        """
        session_id = ev.get("session_id")
        if not session_id or session_id not in self._detached_turns:
            return
        submitted = self._detached_turns.pop(session_id)
        extras = extras or {}
        full, start = self._answer_transcript(extras)
        # The agent that ran THIS turn, carried on its answer. Read off the worker on
        # screen it is the wrong agent whenever the turn belongs elsewhere — which for
        # a detached turn is always.
        context_mode = extras.get("_context_mode") or getattr(
            self.worker._agent, "context_mode", "full")
        result = commit_answer(
            self.store, session_id,
            ev, {**extras, "_full": full, "_turn_start": start},
            submitted_len=extras.get("_submitted_len") or submitted,
            context_mode=context_mode,
        )
        if result is None:
            return
        deferred = result.deferred
        await self._notify(f"“{result.title}” is waiting for your decision." if deferred
                           else f"“{result.title}” finished its background turn.")
        await self._send_sessions_list()
        # The detached twin of the flush after an active turn's answer: jobs that
        # finished into this conversation while it was answering leave together now,
        # against the history this turn just wrote.
        await self._flush_pending_wakes(session_id)
        await self._flush_held_checkin(session_id)

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
        middle = self.history[1:-4]
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
        self.history[1:-4] = summary
        # Slicing across a turn boundary can strand an assistant tool call or its
        # result; strict backends reject that outright.
        self.history[:] = reconcile_tool_pairs(self.history)
        self.pool.bus.record_client_event(self._active_session_id, {
            "type": "context_compact",
            "dropped": len(middle),
            "kept": len(self.history),
            "archived": len(self.history_full),
        })
        return True

    def _apply_setting(self, name: str, *args: Any) -> str:
        """Apply a UI knob to every live conversation, and to the next agent built.

        Both halves are needed. Pushing it to the live workers makes the change take effect
        now; recording it in the pool makes a conversation started afterwards come up on it
        rather than on the default, which matters because an agent is built long after the
        user sets these.

        Returns the first rejection a live worker reported, or "". Nothing to reject when
        none exists yet: the setting is recorded and validated when it is replayed.
        """
        for result in self.pool.apply_setting(name, *args):
            if isinstance(result, str) and result:
                return result
        return ""

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
        counts = backend.message_token_counts(
            self.worker.model, self.history, allow_network=False
        )
        used_tokens = sum(counts)
        if used_tokens >= usable_tokens and len(self.history) > 1:
            # Compaction first: it carries forward what the older turns established,
            # where the front-trim below just forgets them. Only when it cannot run
            # (too short a middle) or the summary still doesn't fit do we drop turns.
            if await self._compact_history():
                counts = backend.message_token_counts(
                    self.worker.model, self.history, allow_network=False
                )
                used_tokens = sum(counts)
                await self._emit_context_usage()
        if used_tokens >= usable_tokens and len(self.history) > 1:
            await self.ws.send(json.dumps({"type": "output",
                "text": "  ⚡ Context budget reached — trimming oldest history…\n"}))
            # Per-message counts computed once; decrement as we pop the front.
            dropped_before = len(self.history)
            total_tokens = used_tokens
            idx = 0
            while total_tokens > usable_tokens and len(self.history) > 1 and idx < len(counts):
                self.history.pop(0)
                total_tokens -= counts[idx]
                idx += 1
            # Front-trimming can orphan a ``{"role": "tool"}`` whose assistant tool_call
            # was popped, which a strict tokenizer rejects — so drop leading
            # tool messages until history starts on a valid turn boundary.
            while self.history and self.history[0].get("role") == "tool":
                self.history.pop(0)
            # `history_full` is deliberately untouched above — this records what the
            # window lost, so the log stays the one complete account of the session.
            self.pool.bus.record_client_event(self._active_session_id, {
                "type": "context_trim",
                "dropped": dropped_before - len(self.history),
                "kept": len(self.history),
                "archived": len(self.history_full),
            })
            await self._emit_context_usage()

    async def _handle_query(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return
        # Writing instead of answering moves past a deferred card: the call keeps its
        # "not run" result, and a late answer to the card goes nowhere.
        if self._pending_interaction is not None:
            prompt_id = (self._pending_interaction.get("prompt") or {}).get("id")
            if prompt_id:
                self._stale_prompt_ids.add(prompt_id)
            self._pending_interaction = None

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

        self.history.append({"role": "user", "content": text})
        self.history_full.append({"role": "user", "content": text})
        self._display_messages.append({"role": "user", "kind": "text", "text": text})
        self.pool.bus.record_client_event(
            self._active_session_id, {"type": "query", "text": text})
        # Save on arrival so a reconnect mid-turn (e.g. while waiting for an edit
        # approval) reloads from disk instead of minting a blank session ID, which
        # would wipe the chat on the frontend.
        self._autosave_session(list(self._display_messages))
        self._submitted_len = len(self.history)
        messages = list(self.history[:-1]) + [{"role": "user", "content": effective_text}]
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
        messages = msg.get("messages")
        if not isinstance(messages, list):
            return
        if self._text_count(messages) < self._text_count(self._display_messages):
            return
        self._display_messages = messages
        # The client says how far the transcript it just sent reaches. Taken on trust
        # the same way the transcript itself is, and only ever forward: a webview that
        # reopened on less than we hold must not shorten the watermark either.
        through = msg.get("through_seq")
        if isinstance(through, int) and through > self._rendered_seq:
            self._rendered_seq = through
        self._autosave_session(list(self._display_messages))

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
            await self._handle_query(msg)
            return
        self.history.append({"role": "user", "content": text})
        # Not added to `history_full` here: the steer comes back inside the turn's own
        # messages when the answer lands, and recording it twice would double it.
        self._display_messages.append({"role": "user", "kind": "text", "text": text})
        self.pool.bus.record_client_event(
            self._active_session_id, {"type": "steer", "text": text})
        self._autosave_session(list(self._display_messages))
        self.worker.submit_steer(text)

    def _session_title(self, session_id: str) -> str:
        """A conversation's name, or "" if it has none (or none saved yet)."""
        try:
            if self._unsaved_session_meta is not None and \
                    self._unsaved_session_meta.id == session_id:
                return self._unsaved_session_meta.title or ""
            return self.store.load_session(session_id).title or ""
        except Exception:
            return ""

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
            server_registry.update(detached=False)
            # Read before the send, not inside it: a getter that raises would otherwise
            # swallow the whole reply, and a client left believing the server is still
            # detached stops guarding something nothing is guarding.
            try:
                level = self.pool.worker_or_detached(
                    self._active_session_id).get_approval_mode()
            except Exception:
                level = "manual"
            try:
                await self.ws.send(json.dumps({
                    "type": "detached", "detached": False, "log": None,
                    "autonomy": level, "sessions": [], "pid": os.getpid(),
                    "setsid": False,
                }))
            except Exception:
                return
            logger.info("detach: this server is this window's again")
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

        session_ids = msg.get("session_ids")
        targeted: list[str] = []
        if isinstance(session_ids, list) and session_ids:
            for sid in session_ids:
                worker = self.pool.get(sid)
                if worker is None:
                    continue
                try:
                    worker.set_approval_mode(autonomy)
                    targeted.append(str(sid))
                except Exception:
                    logger.warning("detach: session %s refused the autonomy level %s",
                                   sid, autonomy, exc_info=True)
        else:
            self._apply_setting("set_approval_mode", autonomy)
            targeted = [sid for sid, _w in self.pool.items()]

        # The agent's stdout is about to become a log file, which is not a tty but is
        # also not proof that nobody is reachable. Saying so plainly keeps the
        # interactive fallbacks — the bare ``input()`` calls behind the CLI's hooks —
        # from being attempted at all.
        for _sid, worker in self.pool.items():
            setter = getattr(worker, "set_non_interactive", None)
            if setter is not None:
                try:
                    setter(True)
                except Exception:
                    logger.warning("detach: session %s could not be marked "
                                   "non-interactive", _sid, exc_info=True)

        info = detach_process(_MIMIR_DIR_WS)
        # The entry already holds the address; this only adds what has changed about
        # the process behind it, so a window that attaches later can say the run is
        # detached and under which level.
        server_registry.update(detached=True, log=info.get("log"), autonomy=autonomy)

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
            worker.resolve_approval(answer["choice"], answer["approved_files"])

    async def _handle_user_question_response(self, msg: dict) -> None:
        if await self._answer_deferred(msg, {"answers": msg.get("answers") or []}):
            return
        worker = self._worker_for_answer(msg)
        if worker is not None:
            # With the card's id: a question whose wait expired refuses its late answer
            # rather than letting it settle the next prompt.
            worker.resolve_question(msg.get("answers"), msg.get("id"))

    async def _resend_deferred_prompt(self) -> None:
        """Put the card a deferred turn of this session waits on back on screen."""
        prompt = (self._pending_interaction or {}).get("prompt")
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
        record = self._pending_interaction
        if not record or not msg_id or msg_id != (record.get("prompt") or {}).get("id"):
            return False
        self._pending_interaction = None
        # Answered once: a second answer to the same card (a copy of it on screen)
        # must not settle the next live prompt either.
        self._stale_prompt_ids.add(msg_id)
        if record.get("kind") == KIND_CALLS:
            # The placeholders leave the record here as they leave the working copy
            # in the loop, so the real results land where they stood.
            take_deferred_calls(self.history_full, record.get("call_ids") or [])
            take_deferred_calls(self.history, record.get("call_ids") or [])
        self._autosave_session(list(self._display_messages))
        self._submitted_len = len(self.history)
        self.worker.submit_resume(record, answer, list(self.history),
                                  session_id=self._active_session_id)
        return True

    async def _handle_batch_review_accept(self, msg: dict) -> None:
        # User accepted all pending file edits — keep them on disk, clear snapshots.
        if self.worker._agent is not None:
            self.worker._agent.approvals._file_snapshots.clear()
            self.worker._agent.approvals.forget_reviewed()
        await self.ws.send(json.dumps({"type": "batch_status", "files": []}))

    async def _handle_batch_review_revert(self, msg: dict) -> None:
        # User wants to undo all pending file edits — restore originals.
        moved: list[str] = []
        if self.worker._agent is not None:
            approvals = self.worker._agent.approvals
            for path, original in list(approvals._file_snapshots.items()):
                if not self._revert_one(path, original):
                    moved.append(os.path.relpath(path))
            approvals._file_snapshots.clear()
            approvals.forget_reviewed()
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
        if approvals.reviewed_matches(path) is False:
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
            snapshots = self.worker._agent.approvals._file_snapshots
            key = next(
                (k for k in snapshots if os.path.normpath(os.path.abspath(k)) == target),
                None,
            )
            if key is not None:
                del snapshots[key]
                self.worker._agent.approvals.forget_reviewed(key)
        self.worker._push_batch_status()

    async def _handle_batch_review_revert_file(self, msg: dict) -> None:
        # Revert a single file — restore its original content.
        file_rel = msg.get("file", "")
        if file_rel and self.worker._agent is not None:
            target = os.path.normpath(os.path.abspath(file_rel))
            snapshots = self.worker._agent.approvals._file_snapshots
            key = next(
                (k for k in snapshots if os.path.normpath(os.path.abspath(k)) == target),
                None,
            )
            if key is not None:
                original = snapshots.pop(key)
                reverted = self._revert_one(key, original)
                self.worker._agent.approvals.forget_reviewed(key)
                if not reverted:
                    await self._report_unreverted([os.path.relpath(key)])
        self.worker._push_batch_status()

    async def _handle_resume_plan(self, msg: dict) -> None:
        choice = msg.get("choice")  # "yes" or "no"
        if choice == "yes":
            self.worker._push_todos()
        else:
            self.worker._clear_todos()
            await self.ws.send(json.dumps({"type": "todo", "items": []}))

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

    def _detach_running_turn(self) -> None:
        """Note where the turn being left stood, then leave it running.

        The turn keeps running: it streams into its own conversation's transcript, and a
        card it is parked on carries the conversation it belongs to, so neither needs the
        user to be looking at it.

        What is recorded is where this conversation's history stood when the turn was
        submitted, so the answer — landing after the user has moved on — is applied to the
        right conversation and can tell its own messages from the prefix it inherited.
        """
        leaving = self._active_session_id
        if leaving and self._running_turn_is_ours():
            self._detached_turns[leaving] = getattr(self, "_submitted_len", 0)

    async def _handle_create_session(self, msg: dict) -> None:
        self._detach_running_turn()
        # Save current session before creating a new one.
        self._autosave_session(list(self._display_messages))
        await self._create_new_session()
        await self._send_sessions_list()

    async def _handle_switch_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        if not target_id or not self.store.session_exists(target_id):
            return
        if target_id == self._active_session_id:
            return
        self._detach_running_turn()
        # Save current before switching.
        self._autosave_session(list(self._display_messages))
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
        wrote no ``exit_code``; a Slurm submission is live when it recorded an id and no
        exit code — neither is a certainty, which is why this reports and does not act.
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
                if os.path.exists(os.path.join(job, "exit_code")):
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
            await self.ws.send(json.dumps({
                "type": "status",
                "text": ("  ⚠ Not deleted — that conversation still has work running: "
                         + "; ".join(running[:3])
                         + (f" (+{len(running) - 3} more)" if len(running) > 3 else "")
                         + ". Stop it first, or delete again to discard it."),
            }))
            await self._send_sessions_list()
            return
        self._delete_refused.discard(target_id)
        was_active = (target_id == self._active_session_id)
        # Its agent goes with it, freeing a slot — and whatever it was queued to run,
        # which would otherwise be admitted into a conversation that is gone.
        await self.pool.close(target_id)
        self.store.delete_session(target_id)
        # Remove the session's sidecar directory (todo_list.md, plan.md, …).
        session_dir = os.path.join(_MIMIR_DIR_WS, "sessions", target_id)
        if os.path.isdir(session_dir):
            import shutil as _shutil
            try:
                _shutil.rmtree(session_dir)
            except OSError:
                pass
        if was_active:
            await self._create_new_session()
        await self.pool.pump()   # a freed slot may admit a conversation that was waiting
        await self._send_sessions_list()

    async def _handle_rename_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        new_title = (msg.get("title") or "").strip()
        if not target_id or not self.store.session_exists(target_id):
            return
        session = self.store.load_session(target_id)
        session.title = new_title
        session.title_custom = True  # a hand-picked title outranks the generated description
        self.store.save_session(session)
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
        nothing in it. ``items`` are ``{label, detail}`` rows.
        """
        await self.ws.send(json.dumps({
            "type":    "command_output",
            "command": command,
            "title":   title,
            "items":   items or [],
            "note":    note,
            "tone":    tone,
        }))

    async def _send_thinking_state(self) -> None:
        """Report the reasoning depth the agent ACTUALLY holds after a change.

        A state message, not a transcript line. The depth already has a control in
        the settings panel, so narrating it in the conversation says twice what the
        panel says once — and the webview replays its stored settings on every
        connect, so that line greeted each session with a depth nobody had just
        chosen. Reporting the agent's own value rather than echoing the requested
        one is also what stops the greeting being wrong: the request is a wish, the
        attribute is the fact.
        """
        agent = getattr(self.worker, "_agent", None)
        depth = getattr(agent, "thinking_depth", None)
        if depth is None:
            return
        label = (THINKING_DEPTH_LABELS[depth]
                 if 0 <= depth < len(THINKING_DEPTH_LABELS) else "")
        await self.ws.send(json.dumps({
            "type": "thinking_depth", "depth": depth, "label": label,
        }))

    async def _send_streaming_state(self) -> None:
        """Report whether the agent is ACTUALLY streaming, after a change.

        Same reasoning as :meth:`_send_thinking_state`: the toggle owns a control in
        the settings panel, and the webview replays its stored value on connect, so a
        transcript line about it was chrome repeated as session-opening noise.
        """
        agent = getattr(self.worker, "_agent", None)
        enabled = getattr(agent, "streaming", None)
        if enabled is None:
            return
        await self.ws.send(json.dumps({
            "type": "streaming", "enabled": bool(enabled),
        }))

    async def _handle_command(self, text: str) -> None:
        """Answer a session command the user typed (never seen by the model).

        Replies go out as ``command_output``, not ``output``. ``output`` is the
        transient tool-activity channel and the webview drops it on purpose — which
        silently swallowed every answer here: ``/memory list`` printed nothing, and
        ``/memory clear`` wiped the store while looking like it had done nothing at
        all. An answer to something the user typed is not activity chatter, so it
        travels on its own channel and is rendered.
        """
        if text.startswith("/mode "):
            mode = text[6:].strip()
            error = self._apply_setting("set_mode", mode)
            if error:
                await self.ws.send(json.dumps({"type": "error", "text": f"  ✗ {error}\n"}))
            else:
                # State, not narration: the mode switcher shows it, so a line in the
                # transcript would only repeat the chrome — and the webview replays
                # its own mode on connect, which made that line session-opening noise.
                await self.ws.send(json.dumps({"type": "mode", "mode": mode}))
        elif text.startswith("/batch "):
            flag = text[7:].strip().lower()
            self._apply_setting("set_batch", flag in ("on", "true", "1", "yes"))
        elif text.startswith("/thinking "):
            flag = text[10:].strip().lower()
            on = flag in ("on", "true", "1", "yes")
            self._apply_setting("set_thinking", on)
            await self._send_thinking_state()
        elif text.startswith("/thinking-depth "):
            arg = text[16:].strip()
            level = thinking_depth_from_label(arg) if not arg.lstrip("-").isdigit() else int(arg)
            if level is None or not 0 <= level < len(THINKING_DEPTH_LABELS):
                usage = f"Usage: /thinking-depth 0-{len(THINKING_DEPTH_LABELS) - 1} ({'|'.join(THINKING_DEPTH_LABELS)})"
                await self.ws.send(json.dumps({"type": "error", "text": usage}))
                return
            self._apply_setting("set_thinking_depth", level)
            await self._send_thinking_state()
        elif text.startswith("/streaming "):
            flag = text[11:].strip().lower()
            on = flag in ("on", "true", "1", "yes")
            self._apply_setting("set_streaming", on)
            await self._send_streaming_state()
        elif text.startswith("/context "):
            mode = text[9:].strip().lower()
            if mode in ("compact", "full"):
                self._apply_setting("set_context_mode", mode)
                # Also what the budget assumes until an agent answers for itself, and
                # what the session is saved with: a mode chosen before the first query
                # must not be forgotten by the one piece of code that sizes the window.
                self._resumed_context_mode = mode
                await self.ws.send(json.dumps({"type": "context_mode", "mode": mode}))
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown context mode: {mode}. Use compact or full."}))
        elif text.startswith("/approvals "):
            # "all" is the spoken form of auto_all — nobody types an underscore.
            raw = text[11:].strip().lower()
            mode = {"all": "auto_all", "auto-all": "auto_all"}.get(raw, raw)
            if mode in ("manual", "auto", "auto_all"):
                self._apply_setting("set_approval_mode", mode)
                await self.ws.send(json.dumps({"type": "approval_mode", "mode": mode}))
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown approval mode: {raw}. Use manual, auto, or all."}))
        elif text.startswith("/enforcement "):
            level = text[13:].strip().lower()
            if level in ("strict", "light", "off"):
                self._apply_setting("set_enforcement", level)
                await self.ws.send(json.dumps({"type": "enforcement", "mode": level}))
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown enforcement level: {level}. Use strict, light, or off."}))
        elif text == "/temperature" or text.startswith("/temperature "):
            raw = text[12:].strip()
            ok, value = parse_temperature(raw) if raw else (True, None)
            if not raw:
                # Bare command: report, change nothing.
                pass
            elif ok:
                self._apply_setting("set_temperature", value)
            else:
                await self.ws.send(json.dumps({"type": "error", "text": (
                    f"Invalid temperature: {raw}. Use a number from "
                    f"{TEMPERATURE_MIN:g} to {TEMPERATURE_MAX:g}, or 'default'.")}))
                return
            # The value the agent holds, not the one asked for (see _send_thinking_state).
            await self.ws.send(json.dumps(
                {"type": "temperature", **self.worker.get_temperature_state()}))
        elif text.startswith("/proxy"):
            # Housekeeping the person running the session may need without asking the
            # model for it. Without these the only way to remove a proxy's runs and
            # optimisation state is to delete a store directory whose path nobody has a
            # reason to know.
            parts = text.split()
            if len(parts) >= 2 and parts[1] == "list":
                # "clean <name>" needs a name, and the only way to learn one was to
                # ask the model to list them — a round trip through an LLM to read a
                # registry file.
                payload = await asyncio.wrap_future(
                    self.worker.call_session_tool("proxy_get", {"op": "proxies"}))
                if payload.get("status") != "ok":
                    await self.ws.send(json.dumps({
                        "type": "error",
                        "text": f"  ✗ {payload.get('error', 'list failed')}\n"}))
                else:
                    entries = payload.get("proxies") or []
                    await self._command_reply(
                        "/proxy list",
                        f"{len(entries)} registered "
                        + ("proxy" if len(entries) == 1 else "proxies")
                        if entries else "No proxies registered",
                        items=[{"label": e.get("name", "?"),
                                "detail": (e.get("description") or "").strip()}
                               for e in entries],
                        tone="ok" if entries else "empty",
                    )
            elif len(parts) >= 3 and parts[1] == "clean":
                name = parts[2]
                payload = await asyncio.wrap_future(
                    self.worker.call_session_tool(
                        "proxy_manage", {"op": "clean", "name": name, "confirm": True}))
                if payload.get("status") != "ok":
                    await self.ws.send(json.dumps({
                        "type": "error",
                        "text": f"  ✗ {payload.get('error', 'clean failed')}\n"}))
                else:
                    # What survived is the part worth showing: this is the exact
                    # question ("I deleted everything and it still remembers") that
                    # made a whole session start from state nobody meant to keep.
                    items = [{"label": "removed",
                              "detail": ", ".join(payload.get("removed") or ["nothing"])}]
                    items += [{"label": "kept", "detail": k}
                              for k in (payload.get("kept") or [])]
                    await self._command_reply(
                        "/proxy clean", f"Cleaned proxy '{name}'",
                        items=items, tone="warn",
                    )
            else:
                await self.ws.send(json.dumps({
                    "type": "error",
                    "text": "Usage: /proxy list  — registered proxies\n"
                            "       /proxy clean <name>  — removes that proxy's runs, "
                            "optimisation state and snapshots, and reports what it "
                            "left behind.\n"}))
        elif text.startswith("/memory"):
            # The memory tools existed but only the model could reach them, so "forget
            # what you learned about this project" was a request rather than an action —
            # and a stale memory keeps being recalled into every prompt until removed.
            parts = text.split()
            sub = parts[1] if len(parts) >= 2 else ""
            if sub == "list":
                payload = await asyncio.wrap_future(
                    self.worker.call_session_tool("memory_list_all", {}))
                entries = payload.get("memory") or []
                await self._command_reply(
                    "/memory list",
                    f"{len(entries)} " + ("memory" if len(entries) == 1 else "memories")
                    if entries else "No memories stored",
                    items=[{"label": e.get("name", "?"),
                            "detail": (e.get("description") or "").strip()}
                           for e in entries],
                    tone="ok" if entries else "empty",
                )
            elif sub == "delete" and len(parts) >= 3:
                payload = await asyncio.wrap_future(
                    self.worker.call_session_tool("memory_delete", {"name": parts[2]}))
                if payload.get("status") != "ok":
                    await self.ws.send(json.dumps({
                        "type": "error",
                        "text": f"  ✗ {payload.get('error', 'delete failed')}\n"}))
                else:
                    await self._command_reply(
                        "/memory delete", "Deleted 1 memory",
                        items=[{"label": parts[2]}], tone="warn")
            elif sub == "clear":
                # Irreversible, and deliberately typed in full by the person whose
                # memory it is. The count is reported so the effect is visible —
                # a wipe that says nothing reads as a wipe that did not happen.
                payload = await asyncio.wrap_future(
                    self.worker.call_session_tool("memory_clear", {}))
                if payload.get("status") != "ok":
                    await self.ws.send(json.dumps({
                        "type": "error",
                        "text": f"  ✗ {payload.get('error', 'clear failed')}\n"}))
                else:
                    n = payload.get("cleared", 0)
                    await self._command_reply(
                        "/memory clear",
                        f"Cleared {n} " + ("memory" if n == 1 else "memories"),
                        note="This cannot be undone.", tone="warn")
            else:
                await self.ws.send(json.dumps({
                    "type": "error",
                    "text": "Usage: /memory list | /memory clear | /memory delete <name>\n"}))
        elif text == "/diag":
            # The chain the events travel, asked rather than inferred. The pump is the
            # only consumer of every worker's queue now, so a chat that goes quiet has
            # several causes that look identical from a chat window: the pump never
            # started, it died, a watermark is swallowing what it delivers, or the agent
            # emitted nothing. These rows tell them apart.
            rows = list(self.pool.bus.diagnostics())
            rows.append({"label": "active session",
                         "detail": str(self._active_session_id)})
            rows.append({"label": "rendered watermark", "detail": str(self._rendered_seq)})
            for sid, count in sorted(self._foreign_withheld.items()):
                rows.append({"label": "withheld as foreign", "detail":
                             f"{count} event(s) stamped {sid} — not this conversation"})
            for sid, worker in self.pool.items():
                rows.append({"label": f"worker {sid[:8]}", "detail":
                             f"queue {worker.out_q.qsize()}, "
                             f"running {getattr(worker, '_query_session_id', None)}, "
                             f"parked {getattr(worker, '_pending_prompt', None) is not None}, "
                             f"deferral {getattr(worker, 'has_deferral', False)}"})
            await self._command_reply("/diag", "Event chain", items=rows)
        elif text == "/cancel":
            cancelled = self.worker.cancel()
            if not cancelled:
                await self._command_reply("/cancel", "Nothing to cancel", tone="empty")
        elif text.startswith("/backend "):
            mode = text[9:].strip().lower()
            if mode in ("ollama", "vllm", "ray"):
                self.worker._agent.set_backend(mode)
                await self._command_reply("/backend", "Backend", items=[{"label": mode}])
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown backend: {mode}. Use ollama, vllm or ray."}))
        else:
            await self.ws.send(json.dumps({"type": "error", "text": f"Unknown command: {text}"}))

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

    async def _handle_list_resources(self, msg: dict) -> None:
        """Serve the attachable-resource list for the webview picker/autocomplete."""
        await self.ws.send(json.dumps({
            "type": "resources",
            "resources": self.worker.resources_snapshot(),
        }))

    async def _handle_toggle_server(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self._apply_setting("set_server_enabled", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_skill(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self._apply_setting("set_skill_enabled", name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_nudge(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self._apply_setting("set_nudge_enabled", name, bool(msg.get("enabled", True)))
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
            "enforcement": self.worker.get_enforcement(),
            "temperature": self.worker.get_temperature_state(),
        }))

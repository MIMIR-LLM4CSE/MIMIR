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
from .ws_worker import _AgentWorker
from ...config import THINKING_DEPTH_LABELS, thinking_depth_from_label

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # annotation-only; avoids the runtime import cycle the methods dodge
    from .session_store import SessionMeta

# The websocket connection handle passed in by the server (websockets library type
# varies by version); kept as an untyped alias so the annotation resolves.
_WS = Any


class _Session:
    """One WebSocket connection — owns a chat history and talks to the worker."""

    def __init__(self, ws: "_WS", worker: _AgentWorker) -> None:
        self.ws = ws
        self.worker = worker
        self.history: list[dict] = []  # LLM history

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

    async def run(self) -> None:
        # Send ready immediately so the webview transitions out of "connecting".
        try:
            await self.ws.send(json.dumps({
                "type": "ready",
                "model": self.worker.model,
                "context_mode": self.worker.get_context_mode(),
                "enforcement": self.worker.get_enforcement(),
            }))
        except Exception:
            return

        # Flush any stale todo/output events left in the queue from a previous session.
        while not self.worker.out_q.empty():
            try:
                self.worker.out_q.get_nowait()
            except Exception:
                break

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

        drain_task = asyncio.create_task(self._drain_loop())
        try:
            async for raw in self.ws:
                await self._handle(raw)
        except Exception:
            pass
        finally:
            if self._summary_task is not None:
                self._summary_task.cancel()
            drain_task.cancel()
            try:
                await drain_task
            except (asyncio.CancelledError, Exception):
                pass

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
                "sessions": [s.to_dict() for s in sessions],
            }))
        except Exception:
            pass

    async def _create_new_session(self) -> None:
        session = self.store.new_session()
        # Don't save to disk yet — wait until there are messages to avoid
        # accumulating empty "New session" entries on every reconnect.
        self._active_session_id = session.id
        self.worker.active_session_id = session.id
        _write_active_session(session.id)
        self.history = []
        self._display_messages = []
        self.worker.load_agent_state({})
        self.worker.reset_session_guards()  # fresh session → drop grants + repeat guard
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
        await self._emit_context_usage()

    async def _load_session(self, session_id: str) -> None:
        session = self.store.load_session(session_id)
        switching = self.worker.active_session_id not in (None, session.id)
        self._active_session_id = session.id
        self._unsaved_session_meta = None  # switching to a persisted session
        self.worker.active_session_id = session.id
        if switching:  # different session → drop prior grants + repeat guard
            self.worker.reset_session_guards()
        _write_active_session(session.id)
        self.history = list(session.llm_history)
        self._display_messages = list(session.display_messages)
        self.worker.load_agent_state({"carry_context": session.carry_context})

        # If session had todos, restore them to disk so agent picks them up.
        if session.todos:
            self._restore_todos(session.todos, getattr(session, 'todo_deps', None))
            # Only offer to resume if there are incomplete tasks remaining.
            pending_todos = [t for t in session.todos if not t.get("done")]
            try:
                await self.ws.send(json.dumps({
                    "type": "session_loaded",
                    "session_id": session.id,
                    "title": session.title,
                    "display_messages": session.display_messages,
                    "todos": [] if pending_todos else session.todos,
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
                }))
            except Exception:
                pass
        await self._emit_context_usage()

    def _restore_todos(self, todos: list[dict], todo_deps: list | None = None) -> None:
        """Write todo items back to the session-scoped todo file.

        Also restores the todo_deps.json sidecar when deps are provided.
        """
        try:
            todo_file = _todo_file_for_session(self._active_session_id)
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
        except Exception:
            pass

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
            session.display_messages = list(display_messages)
            session.carry_context = agent_state.get("carry_context", {})
            session.todos = self.worker._load_todos()
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
            # Refresh unless the stored description already covers exactly this
            # transcript and came from the current generator. A provisional one
            # (the query fallback) never counts as fresh, so it keeps retrying.
            fresh_enough = (
                session.summary
                and session.summary_version == SUMMARY_VERSION
                and session.summary_msgs >= len(messages)
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
            fresh.summary_msgs = len(messages)
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
        mode = self.worker.get_context_mode()
        total, reserved, _, _ = context_budget_for(self.worker.model, mode)
        return total, reserved

    async def _emit_context_usage(self) -> None:
        """Push a context_usage event to the WS client (best-effort, never raises)."""
        try:
            total, reserved = self._ctx_budget()
            # allow_network=False: this runs on the WS event loop, which must not
            # block on a tokenize round-trip. Already-counted messages (tokenized
            # by the agent worker during the query) hit the shared cache for exact
            # numbers; the rest fall back to the heuristic.
            history_used = get_backend().count_messages_tokens(
                self.worker.model, self.history, allow_network=False
            )
            # Include the fixed per-call overhead (system prompt + tools schema).
            # Without it the bar shows only the conversation and hides the ~8–12k
            # tokens that actually push the prompt over the model window.
            overhead = self.worker.context_overhead_tokens()
            used = history_used + overhead
            await self.ws.send(json.dumps({
                "type": "context_usage",
                "used_tokens": used,
                "total_tokens": total,
                "reserved_tokens": reserved,
                "overhead_tokens": overhead,
            }))
        except Exception:
            pass

    # ── Drain loop ────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        """Forward worker output queue to the WS client."""
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

        while True:
            events = self.worker.drain()
            for ev in events:
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
                try:
                    await self.ws.send(json.dumps(ev, default=str))
                except Exception:
                    return
                if ev.get("type") == "error":
                    # A turn failed (often a context-overflow 400). No `answer`
                    # event will follow, so refresh the context bar here too —
                    # otherwise the bar keeps showing the pre-failure usage and
                    # never reflects the overflow that caused the error.
                    await self._emit_context_usage()
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
                if ev.get("type") == "answer":
                    # In full-context mode keep the structured transcript (user
                    # turn + assistant tool_calls + tool results + final answer,
                    # with chain-of-thought already stripped) so the model recalls
                    # the tools it ran — the same behaviour as the CLI chat loop.
                    # Falls back to the flattened final answer otherwise (or when
                    # no structured transcript is available yet).
                    full = self.worker.full_history()
                    context_mode = getattr(self.worker._agent, "context_mode", "full")
                    if full is not None and context_mode == "full":
                        self.history = full
                    else:
                        self.history.append({"role": "assistant", "content": ev.get("text", "")})
                    self._display_messages.append({
                        "role": "agent",
                        "kind": "text",
                        "text": ev.get("text", ""),
                    })
                    self._autosave_session(list(self._display_messages))
                    # Describe the session once the turn has landed: the answer is
                    # part of the transcript, so the description says what was done
                    # and not just what was asked — and the model call no longer
                    # competes with the query the user is waiting on.
                    self._schedule_summary_refresh()
                    await self._emit_context_usage()
                    # Push the batch-review bar state at answer time.  Sending
                    # directly here (not via out_q) means we read the snapshot
                    # dict *after* any batch_review_accept that arrived while the
                    # agent was running has already cleared it.
                    try:
                        files = self.worker._build_batch_status()
                        await self.ws.send(json.dumps({"type": "batch_status", "files": files}))
                    except Exception:
                        pass
                if ev.get("type") == "job_complete":
                    await self._handle_job_complete(ev)
            await asyncio.sleep(0.005)
            await _check_and_push_todos()

    # Maps an inbound WS message "type" to the _Session handler method that serves it.
    # Replaces the former 13-branch ``if mtype == …`` chain in _handle.
    _MSG_HANDLERS: dict[str, str] = {
        "query": "_handle_query",
        "steer": "_handle_steer",
        "approval_response": "_handle_approval_response",
        "continue_response": "_handle_continue_response",
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

        Enriched from a proxy results summary (verdict/best/next_step) when present,
        else a generic completion + diagnose-on-crash instruction.
        """
        job_key = ev.get("job_key", "?")
        state   = ev.get("state", "done")
        server  = ev.get("server")
        summary = ev.get("summary") if isinstance(ev.get("summary"), dict) else {}
        if state == "crashed":
            how = ("proxy_eval_status(op='log', tail=100)" if server == "proxy"
                   else "the job's Slurm logs (slurm_job_status / sacct)")
            return (f"Background job '{job_key}' crashed. Inspect the failure via "
                    f"{how} and decide how to proceed.")
        verdict   = summary.get("verdict")
        best      = summary.get("best") or {}
        next_step = summary.get("next_step")
        parts = [f"Background job '{job_key}' finished."]
        if verdict:
            parts.append(f"verdict={verdict}")
        if isinstance(best, dict) and best.get("primary_value") is not None:
            parts.append(f"best {summary.get('primary_metric', 'primary')}="
                         f"{best.get('primary_value')}")
        head = " ".join(parts)
        tail = (next_step or "Review proxy_eval_status(op='results') and continue "
                "the optimization loop, or summarize if converged.")
        return f"{head} {tail}"

    async def _handle_job_complete(self, ev: dict) -> None:
        """Notify the user and auto-resume the agent when a background job finishes.

        The event was already forwarded to the client (notification) by the drain
        loop. Here we synthesize a wake and enqueue it through the normal query path
        so the agent resumes with full session history; the serial query loop makes
        it queue behind any turn currently in flight.
        """
        wake = self._wake_text(ev)
        # Show a distinct system-style note rather than a fake user bubble.
        self._display_messages.append({
            "role": "system", "kind": "text",
            "text": f"🔔 {wake}",
        })
        self.history.append({"role": "user", "content": wake})
        self._autosave_session(list(self._display_messages))
        self.worker.submit_query(wake, list(self.history))

    async def _handle_query(self, msg: dict) -> None:
        text = (msg.get("text") or "").strip()
        if not text:
            return

        # Pre-query budget check: if history already fills the usable window,
        # trim the oldest messages from the front so the new query fits.
        # We do NOT call compact_history here (LLM call from the event loop
        # is unsafe while the worker thread owns the agent). Simple truncation
        # is fast, deterministic, and keeps the most recent context.
        total, reserved = self._ctx_budget()
        # The system prompt and tools schema ride on every call, so history only
        # gets what is left after them. Leaving the overhead out here (while the
        # context bar includes it) let the bar read past 100% without the trim
        # ever firing — the prompt overflowed while this check said it fit.
        usable_tokens = max(1, total - reserved - self.worker.context_overhead_tokens())
        # allow_network=False: never block the WS event loop on tokenize.
        backend = get_backend()
        counts = backend.message_token_counts(
            self.worker.model, self.history, allow_network=False
        )
        used_tokens = sum(counts)
        if used_tokens >= usable_tokens and len(self.history) > 1:
            await self.ws.send(json.dumps({"type": "output",
                "text": "  ⚡ Context budget reached — trimming oldest history…\n"}))
            # Per-message counts computed once; decrement as we pop the front.
            total_tokens = used_tokens
            idx = 0
            while total_tokens > usable_tokens and len(self.history) > 1 and idx < len(counts):
                self.history.pop(0)
                total_tokens -= counts[idx]
                idx += 1
            # Front-trimming can leave an orphaned tool result at the head (a
            # ``{"role": "tool"}`` whose preceding assistant tool_call was popped),
            # which strict tokenizers (e.g. Mistral) reject. Drop any such leading
            # tool messages so history always starts on a valid turn boundary.
            while self.history and self.history[0].get("role") == "tool":
                self.history.pop(0)
            await self._emit_context_usage()

        # Resolve @<uri> resource mentions on the worker's loop (where the MCP
        # sessions live), then submit the augmented text to the model. History and
        # display keep the RAW `text`, so the attachment is per-turn (Claude/Copilot
        # style). Caveat: in full-context mode the augmented user message persists in
        # the worker's `_last_full_messages`; acceptable for MVP.
        effective_text = text
        try:
            effective_text, attached_uris = await asyncio.wrap_future(
                self.worker.resolve_resources(text)
            )
        except Exception:
            attached_uris = []
        if attached_uris:
            await self.ws.send(json.dumps({
                "type": "output",
                "text": "📎 Attached: " + ", ".join(attached_uris) + "\n",
            }))

        self.history.append({"role": "user", "content": text})
        self._display_messages.append({"role": "user", "kind": "text", "text": text})
        # Save as soon as a query arrives so a reconnect during long-running
        # agent turns (e.g. while waiting for an edit approval) can reload
        # the session from disk and won't create a new blank session ID,
        # which would wipe the chat on the frontend.
        self._autosave_session(list(self._display_messages))
        self.worker.submit_query(effective_text, list(self.history[:-1]) + [{"role": "user", "content": effective_text}])

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
        if not self.worker.is_busy():
            await self._handle_query(msg)
            return
        self.history.append({"role": "user", "content": text})
        self._display_messages.append({"role": "user", "kind": "text", "text": text})
        self._autosave_session(list(self._display_messages))
        self.worker.submit_steer(text)

    async def _handle_approval_response(self, msg: dict) -> None:
        self.worker.resolve_approval(msg.get("choice", "n"), msg.get("approved_files"))

    async def _handle_continue_response(self, msg: dict) -> None:
        self.worker.resolve_continue(msg.get("choice", "n"))

    async def _handle_user_question_response(self, msg: dict) -> None:
        self.worker.resolve_question(msg.get("answers"))

    async def _handle_batch_review_accept(self, msg: dict) -> None:
        # User accepted all pending file edits — keep them on disk, clear snapshots.
        if self.worker._agent is not None:
            self.worker._agent.approvals._file_snapshots.clear()
        await self.ws.send(json.dumps({"type": "batch_status", "files": []}))

    async def _handle_batch_review_revert(self, msg: dict) -> None:
        # User wants to undo all pending file edits — restore originals.
        if self.worker._agent is not None:
            for path, original in list(self.worker._agent.approvals._file_snapshots.items()):
                abs_path = os.path.abspath(path)
                try:
                    if original is None:
                        os.remove(abs_path)
                    else:
                        with open(abs_path, "w", encoding="utf-8") as fh:
                            fh.write(original)
                except OSError:
                    pass
            self.worker._agent.approvals._file_snapshots.clear()
        await self.ws.send(json.dumps({"type": "batch_status", "files": []}))

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
                abs_key = os.path.abspath(key)
                try:
                    if original is None:
                        os.remove(abs_key)
                    else:
                        with open(abs_key, "w", encoding="utf-8") as fh:
                            fh.write(original)
                except OSError:
                    pass
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

    async def _handle_create_session(self, msg: dict) -> None:
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
        # Save current before switching.
        self._autosave_session(list(self._display_messages))
        await self._load_session(target_id)
        await self._send_sessions_list()

    async def _handle_delete_session(self, msg: dict) -> None:
        target_id = (msg.get("session_id") or "").strip()
        if not target_id:
            return
        was_active = (target_id == self._active_session_id)
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
        await self._send_sessions_list()

    async def _handle_command_msg(self, msg: dict) -> None:
        await self._handle_command((msg.get("text") or "").strip())

    async def _handle_command(self, text: str) -> None:
        if text.startswith("/mode "):
            mode = text[6:].strip()
            error = self.worker.set_mode(mode)
            if error:
                await self.ws.send(json.dumps({"type": "error", "text": f"  ✗ {error}\n"}))
            else:
                await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Mode set to {mode}\n"}))
        elif text.startswith("/batch "):
            flag = text[7:].strip().lower()
            self.worker.set_batch(flag in ("on", "true", "1", "yes"))
            await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Batch mode {flag}\n"}))
        elif text.startswith("/thinking "):
            flag = text[10:].strip().lower()
            self.worker.set_thinking(flag in ("on", "true", "1", "yes"))
            await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Thinking {'on' if flag in ('on','true','1','yes') else 'off'}\n"}))
        elif text.startswith("/thinking-depth "):
            arg = text[16:].strip()
            level = thinking_depth_from_label(arg) if not arg.lstrip("-").isdigit() else int(arg)
            if level is None or not 0 <= level < len(THINKING_DEPTH_LABELS):
                usage = f"Usage: /thinking-depth 0-{len(THINKING_DEPTH_LABELS) - 1} ({'|'.join(THINKING_DEPTH_LABELS)})"
                await self.ws.send(json.dumps({"type": "error", "text": usage}))
                return
            self.worker.set_thinking_depth(level)
            await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Thinking depth: {THINKING_DEPTH_LABELS[level]}\n"}))
        elif text.startswith("/streaming "):
            flag = text[11:].strip().lower()
            self.worker.set_streaming(flag in ("on", "true", "1", "yes"))
            await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Streaming {'on' if flag in ('on','true','1','yes') else 'off'}\n"}))
        elif text.startswith("/context "):
            mode = text[9:].strip().lower()
            if mode in ("compact", "full"):
                self.worker.set_context_mode(mode)
                await self.ws.send(json.dumps({"type": "context_mode", "mode": mode}))
                await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Context mode set to {mode}\n"}))
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown context mode: {mode}. Use compact or full."}))
        elif text.startswith("/enforcement "):
            level = text[13:].strip().lower()
            if level in ("strict", "light", "off"):
                self.worker.set_enforcement(level)
                await self.ws.send(json.dumps({"type": "enforcement", "mode": level}))
                await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Enforcement set to {level}\n"}))
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown enforcement level: {level}. Use strict, light, or off."}))
        elif text == "/cancel":
            cancelled = self.worker.cancel()
            if not cancelled:
                await self.ws.send(json.dumps({"type": "output", "text": "  (nothing to cancel)\n"}))
        elif text.startswith("/backend "):
            mode = text[9:].strip().lower()
            if mode in ("ollama", "vllm"):
                self.worker._agent.set_backend(mode)
                await self.ws.send(json.dumps({"type": "output", "text": f"  ✓ Backend set to {mode}\n"}))
            else:
                await self.ws.send(json.dumps({"type": "error", "text": f"Unknown backend: {mode}. Use ollama or vllm."}))
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
            self.worker.set_server_enabled(name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_skill(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.worker.set_skill_enabled(name, bool(msg.get("enabled", True)))
        await self._send_toggles()

    async def _handle_toggle_nudge(self, msg: dict) -> None:
        name = msg.get("name")
        if name:
            self.worker.set_nudge_enabled(name, bool(msg.get("enabled", True)))
        await self._send_toggles()

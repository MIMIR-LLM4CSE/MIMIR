"""The background agent worker for the WebSocket server.

``_AgentWorker`` runs MimirAgent in a dedicated thread with its own asyncio event
loop so blocking approval prompts never freeze the WebSocket event loop. It talks
to the WS layer (``_Session`` in ``ws_session``) exclusively through thread-safe
queues. Split out of ``ws_server.py``; see that module's docstring for the wire
protocol.
"""

from __future__ import annotations

# Import the shared runtime FIRST so its cwd bootstrap runs before config.constants
# (and the backend factory) capture the workspace root at import time.
from ._ws_runtime import (
    _ROUTER,
    _todo_file_for_session,
    augment_query_with_resources,
)

import asyncio
import concurrent.futures
import json
import os
import queue as _queue
import threading
import uuid
from typing import Any

from ... import human_pause


class _AgentWorker:
    """Runs MimirAgent in a dedicated background thread with its own event loop.

    Thread-safe queues carry events to the WS layer and approval responses back.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self._loop: asyncio.AbstractEventLoop | None = None
        self._agent: Any = None
        self._ready = threading.Event()
        self._error: Exception | None = None

        # Queues for cross-thread communication.
        self.out_q: _queue.Queue[dict] = _queue.Queue()   # agent → WS
        self._approval_q: _queue.Queue[dict] = _queue.Queue()  # WS → approval shim
        self._continue_q: _queue.Queue[dict] = _queue.Queue()  # WS → continue shim
        self._question_q: _queue.Queue[dict] = _queue.Queue()  # WS → question shim
        self._query_q: _queue.Queue[dict | None] = _queue.Queue()  # WS → query loop
        self._steer_q: _queue.Queue[str] = _queue.Queue()  # WS → running agent (mid-run steering)
        # Event signalled whenever a new item is placed on _query_q so the
        # background loop wakes up immediately instead of waiting out the poll interval.
        self._query_event = threading.Event()

        self._current_task: asyncio.Task | None = None
        self.active_session_id: str | None = None  # kept in sync by _Session
        # Session a running query belongs to, captured when it starts. One worker
        # serves every session, so events must carry the session they were produced
        # for or a turn that outlives a switch lands in the wrong conversation.
        self._query_session_id: str | None = None
        # Background-job watchers: job_key -> asyncio.Task polling a detached run to
        # completion. Registered by the agent loop via _register_bg_job (below).
        self._bg_jobs: dict[str, asyncio.Task] = {}

        self._thread = threading.Thread(target=self._main, name="mimir-worker", daemon=True)
        self._thread.start()
        # Wait for server connections. The timeout is long to accommodate vLLM
        # cold-start (model load + torch.compile can exceed 3 min); a separate
        # backend-health poll runs first and surfaces progress to the client.
        timeout = int(os.environ.get("MIMIR_INIT_TIMEOUT", "600"))
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(
                f"MimirAgent worker failed to initialise within {timeout} s"
            )
        if self._error:
            raise self._error

    # ── Background thread ─────────────────────────────────────────────────────

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._setup())
            if self._error is None:
                loop.run_until_complete(self._query_loop())
        finally:
            loop.close()

    async def _wait_for_backend(self) -> None:
        """Poll the LLM backend health endpoint until it responds or we time out.

        Supports:
        - Ollama: GET {OLLAMA_BASE_URL}/api/tags  → 200
        - vLLM:   GET {VLLM_BASE_URL}/health      → 200

        Emits ``{"type": "output", "text": "..."}`` progress messages every
        10 s so the client can show a spinner during slow vLLM cold-starts.
        Controlled by env vars:
        - MIMIR_BACKEND_TIMEOUT  (seconds, default 600)
        - MIMIR_BACKEND_POLL_INTERVAL (seconds, default 5)
        """
        import urllib.request as _req
        import urllib.error as _uerr

        try:
            from ...config.models import LLM_BACKEND, VLLM_BASE_URL
        except ImportError:
            LLM_BACKEND = os.environ.get("LLM_BACKEND", "vllm")
            VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")

        if LLM_BACKEND == "vllm":
            base = os.environ.get("VLLM_BASE_URL", VLLM_BASE_URL).rstrip("/")
            health_url = f"{base}/health"
        else:
            base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            health_url = f"{base}/api/tags"

        timeout = int(os.environ.get("MIMIR_BACKEND_TIMEOUT", "600"))
        poll = float(os.environ.get("MIMIR_BACKEND_POLL_INTERVAL", "5"))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_progress = loop.time()
        progress_interval = 10.0

        def _check() -> bool:
            try:
                with _req.urlopen(health_url, timeout=4) as r:
                    return r.status == 200
            except _uerr.HTTPError as e:
                # 403/401 means server is up but requires auth — treat as ready
                return e.code in (401, 403)
            except (_uerr.URLError, OSError):
                return False

        self.out_q.put({"type": "output", "text": f"⏳ Waiting for LLM backend ({LLM_BACKEND}) at {health_url} …\n"})

        while True:
            ready = await loop.run_in_executor(None, _check)
            if ready:
                self.out_q.put({"type": "output", "text": "✅ LLM backend is ready.\n"})
                return

            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"LLM backend did not become ready within {timeout} s "
                    f"(health URL: {health_url})"
                )
            if now - last_progress >= progress_interval:
                elapsed = int(now - (deadline - timeout))
                self.out_q.put({"type": "output", "text": f"⏳ Still waiting for LLM backend … ({elapsed}s elapsed)\n"})
                last_progress = now

            await asyncio.sleep(poll)

    async def _setup(self) -> None:
        try:
            await self._wait_for_backend()
            try:
                from ...agent_core import MimirAgent
            except ImportError:
                from mimir.client.agent_core import MimirAgent
            try:
                from ...extensions import all_servers
            except ImportError:
                from mimir.client.extensions import all_servers

            agent = MimirAgent(model=self.model)
            for name, script in all_servers().items():
                await agent.connect_server(name, script)
            agent.seed_classification_from_caps()

            # Patch approval to route through WS.
            agent._request_tool_approval = self._approval_shim
            # Allow the agent loop to ask the user to extend a long run, routed
            # through the same WS request/response shim pattern as approvals.
            agent.allow_continue_prompt = True
            agent._request_continue = self._continue_shim
            # Route agent clarification questions (the ``ask_user_question`` tool,
            # delivered via MCP elicitation) through the same WS shim pattern.
            agent._request_user_question = self._question_shim
            # Route out-of-workspace file-access approval through the same UI.
            agent._request_path_approval = self._path_approval_shim
            # Let the loop pick up messages the user types WHILE the agent is
            # working (chat-while-busy steering); drained at each step boundary.
            agent._poll_steer = self._drain_steer_q
            # Let the loop detach a long run: the agent ends its turn and this
            # worker watches the run to completion, then notifies + auto-resumes.
            agent._register_background_job = self._register_bg_job

            self._agent = agent
            self.out_q.put({"type": "ready", "model": self.model})
        except Exception as exc:
            self._error = exc
        finally:
            self._ready.set()
            # Pre-warm the model in the background so VRAM is ready for the first
            # real query.  Runs after _ready so it never blocks the WS handshake.
            if self._agent is not None:
                asyncio.get_event_loop().create_task(self._prewarm_model())

    async def _prewarm_model(self) -> None:
        import urllib.request as _urllib_req
        import json as _json_pw
        import os as _os
        _base = _os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        _url = f"{_base}/api/generate"
        _body = _json_pw.dumps({
            "model": self.model,
            "prompt": "",
            "stream": False,
            "keep_alive": "60m",
        }).encode()
        def _call():
            req = _urllib_req.Request(_url, data=_body,
                                      headers={"Content-Type": "application/json"})
            with _urllib_req.urlopen(req, timeout=120):
                pass
        try:
            await asyncio.get_event_loop().run_in_executor(None, _call)
        except Exception:
            pass  # Non-fatal — first real query will trigger load.

    async def _query_loop(self) -> None:
        while True:
            # Wait until a query (or shutdown sentinel) arrives.
            # Use a short asyncio sleep + threading.Event combo so we yield to
            # the event loop while still waking immediately when work arrives.
            try:
                item = self._query_q.get_nowait()
            except _queue.Empty:
                # Block the thread for up to 0.02 s, then re-check.
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._query_event.wait(0.02)
                )
                self._query_event.clear()
                continue

            if item is None:
                return  # shutdown sentinel

            self._current_task = asyncio.current_task()
            self._query_session_id = self.active_session_id
            await self._run_query(item)
            self._current_task = None
            self._query_session_id = None

    async def _run_query(self, item: dict) -> None:
        query = item.get("text", "")
        history = item.get("history", [])

        # Clear cancel flag from any previous cancellation before starting.
        if self._agent is not None:
            self._agent._cancel_flag.clear()
        # Drop any steer messages left over from a finished/cancelled run so they
        # can't bleed into this query (mirrors the cancel-flag clear above).

        tid = threading.get_ident()
        _ROUTER.register(tid, lambda text: self.out_q.put({"type": "output", "text": text}))

        cancelled = False
        try:
            def _token_cb(delta: str) -> None:
                self.out_q.put({"type": "token", "text": delta})

            # Reasoning text of the streaming block, kept so its size can be reported
            # in tokens when the block closes. Counted here, not in the webview, so the
            # number comes from the same tokenizer as the context bar.
            think_buf: list[str] = []

            def _think_token_cb(delta: str) -> None:
                think_buf.append(delta)
                self.out_q.put({"type": "thinking", "text": delta})

            def _think_start_cb() -> None:
                think_buf.clear()
                self.out_q.put({"type": "thinking_start"})

            def _think_end_cb() -> None:
                text = "".join(think_buf)
                think_buf.clear()
                self.out_q.put({"type": "thinking_end", "tokens": self._count_tokens(text)})

            # Structured events (status/tool_call/tool_result/diff) go straight
            # onto out_q as event dicts — no stdout round-trip. The drain/send
            # loop already forwards any dict via json.dumps(ev).
            def _event_cb(ev: dict) -> None:
                self.out_q.put(ev)

            task = asyncio.ensure_future(
                self._agent.run(
                    query=query,
                    history=history,
                    streaming=self._agent.streaming,
                    thinking=self._agent.thinking,
                    token_callback=_token_cb,
                    think_token_callback=_think_token_cb,
                    think_start_callback=_think_start_cb,
                    think_end_callback=_think_end_cb,
                    event_callback=_event_cb,
                )
            )
            self._current_task = task
            answer = await task
        except asyncio.CancelledError:
            cancelled = True
            answer = "[Cancelled]"
        except Exception as exc:
            answer = f"[Error] {exc}"
            self.out_q.put({"type": "error", "text": str(exc)})
        finally:
            self._current_task = None
            _ROUTER.unregister(tid)
            # Always clear the cancel flag after the query ends so it does not
            # bleed into the next query (race: user cancels then immediately
            # sends a new message before _run_query starts again).
            if self._agent is not None:
                self._agent._cancel_flag.clear()

        # Push current todo state.
        self._push_todos()
        self.out_q.put({"type": "answer", "text": answer, "cancelled": cancelled})
        # batch_status is sent from the drain loop's answer handler instead, so it
        # reflects the current snapshot — a batch_status queued here would lose the
        # race with an in-flight batch_review_accept.

    def _build_batch_status(self) -> list[dict]:
        """Compute per-file accumulated diffs from original → current for every
        snapshotted file.  Returns a list of diff entries suitable for the
        ``batch_status`` WebSocket message.
        """
        import difflib as _difflib

        agent = self._agent
        if agent is None:
            return []
        snapshots = agent.approvals._file_snapshots
        files: list[dict] = []
        for path, original in list(snapshots.items()):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    current = fh.read()
            except OSError:
                current = ""
            before_lines = original.splitlines(keepends=True) if original is not None else []
            after_lines  = current.splitlines(keepends=True)
            if before_lines == after_lines:
                continue
            rel = os.path.relpath(path)
            diff_lines = list(_difflib.unified_diff(
                before_lines, after_lines,
                fromfile="a/" + rel, tofile="b/" + rel, n=3,
            ))
            if diff_lines:
                entry: dict = {"file": rel, "patch": "".join(diff_lines)}
                if original is None:
                    entry["is_new"] = True
                files.append(entry)
        return files

    def _push_batch_status(self) -> None:
        try:
            files = self._build_batch_status()
        except Exception:
            files = []
        self.out_q.put({"type": "batch_status", "files": files})

    def _push_todos(self) -> None:
        try:
            items = self._load_todos()
            if items:
                self.out_q.put({"type": "todo", "items": items})
        except Exception:
            pass

    def _load_todos(self) -> list:
        try:
            try:
                from ...prompt.system_prompt import _load_todo_items
            except ImportError:
                from mimir.client.prompt.system_prompt import _load_todo_items
            return _load_todo_items(_todo_file_for_session(self.active_session_id))
        except Exception:
            return []

    def _clear_todos(self) -> None:
        """Wipe the active session's todo file."""
        try:
            todo_file = _todo_file_for_session(self.active_session_id)
            if os.path.exists(todo_file):
                open(todo_file, "w").close()
        except Exception:
            pass

    # ── Approval shim (called sync from agent's async call chain) ─────────────

    def _await_response(self, q: "_queue.Queue[dict]") -> dict | None:
        """Block until a WS response lands on ``q`` — with no wall-clock timeout.

        An unanswered approval/continue/question must keep the agent *parked*: it
        must never silently proceed just because the user was slow to respond.
        So we wait indefinitely instead of timing out. To stay responsive to the
        Stop button, we poll in short slices and bail the moment the agent's
        cancel flag is set (from the WS thread), returning ``None`` for cancelled.

        This is the single seam every WS prompt (approval, out-of-workspace path,
        continue, question) blocks on, so it is where the wait is marked as *human*
        time — excluded from the tool-call timeout budget it sits inside.
        """
        with human_pause.human_pause():
            while True:
                agent = self._agent
                if agent is not None and agent._cancel_flag.is_set():
                    return None
                try:
                    return q.get(timeout=0.25)
                except _queue.Empty:
                    continue

    def _approval_shim(
        self, tool_name: str, arguments: dict, max_attempts: int = 3
    ) -> tuple[bool, str]:
        """Sync approval — blocks background thread until WS client responds.

        The WS event loop (main thread) is unaffected; it forwards the approval
        prompt to the client and puts the response in _approval_q.
        """
        from ...context.capabilities import (
            label_for, preview_spec, reversibility_of,
        )
        from ...tool_execution.tool_status_messages import shorten_display_args
        from .file_preview import build_preview_diffs

        agent = self._agent
        server_name = agent.tool_owner.get(tool_name, "unknown")
        risk = agent.approvals.describe_risk(tool_name)
        scope_label = agent.approvals._approval_scope_label(tool_name, server_name, arguments)
        # Canonical human label ("Proxy exec: run") so the card header matches the
        # tool-activity row instead of showing a bare op name; None when the tool
        # declares no template, and the client falls back to server·tool.
        label = label_for(
            tool_name,
            shorten_display_args(tool_name, arguments or {}, agent.tool_caps),
            agent.tool_caps,
        )

        # A declared diff-preview spec marks a previewable file mutation: auto-approve
        # (batch review + /undo cover it) and send a live card with the reconstructed
        # diff. No spec — including foreign-server writers — means the normal prompt.
        if preview_spec(tool_name, agent.tool_caps) is not None:
            diffs = build_preview_diffs(tool_name, arguments, agent.tool_caps)
            if diffs:
                self.out_q.put({"type": "file_progress", "diffs": diffs})
            return True, ""

        req_id = str(uuid.uuid4())
        payload: dict = {
            "type": "approval",
            "id": req_id,
            "tool": tool_name,
            "server": server_name,
            "args": arguments,
            "risk": risk,
            # The declared undo level, so the card can show severity from a value
            # instead of keyword-sniffing the risk sentence for "destructive".
            "reversibility": reversibility_of(tool_name, agent.tool_caps),
            "scope": scope_label,
            "label": label,
        }
        self.out_q.put(payload)

        # Block background thread (not WS event loop) until the client responds.
        # No timeout: an unanswered prompt keeps the agent parked (Stop cancels).
        response = self._await_response(self._approval_q)
        if response is None:
            return False, "cancelled"

        choice = response.get("choice", "n")

        if choice == "a":
            scope = agent.approvals._approval_scope(tool_name, server_name, arguments)
            agent.approvals.approved_scopes.add(scope)
            return True, f"approved always ({scope_label})"
        if choice == "y":
            return True, "approved once"
        return False, "denied"

    def _path_approval_shim(
        self, abspath: str, tool_name: str, arguments: dict | None = None
    ) -> tuple[bool, bool]:
        """Out-of-workspace access prompt — reuses the approval UI (y/n/a).

        Carries the same descriptive payload as a normal approval (canonical label,
        the call's real arguments, the owning server) so the card reads like any
        other tool card; ``oow_path`` is what makes it an out-of-workspace prompt and
        names the offending path. Returns (approved, always). Blocks the worker thread
        on the approval queue like ``_approval_shim``; the engine's out-of-workspace
        gate records the grant.
        """
        from ...context.capabilities import IRREVERSIBLE, label_for
        from ...tool_execution.tool_status_messages import shorten_display_args

        agent = self._agent
        arguments = arguments or {}
        req_id = str(uuid.uuid4())
        self.out_q.put({
            "type": "approval",
            "id": req_id,
            "tool": tool_name,
            "server": agent.tool_owner.get(tool_name, "filesystem"),
            "args": arguments or {"path": abspath},
            "risk": agent.approvals.describe_risk(tool_name),
            # Reaching outside the workspace is irreversible by situation, not by tool:
            # whatever the tool's own level, nothing here can undo a write landing
            # outside the sandbox, so the card must not soften it to the tool's rating.
            "reversibility": IRREVERSIBLE,
            "scope": f"this path ({os.path.basename(abspath)})",
            # Short label in the header; the card renders `oow_path` beneath it as an
            # explicit "outside workspace" line. The absolute path is the decision
            # being made, so it is never the thing that gets shortened.
            "label": label_for(
                tool_name,
                shorten_display_args(tool_name, arguments or {}, agent.tool_caps),
                agent.tool_caps,
            ),
            "oow_path": abspath,
        })
        # No timeout: keep the agent parked until answered (Stop cancels).
        response = self._await_response(self._approval_q)
        if response is None:
            return (False, False)
        choice = response.get("choice", "n")
        if choice == "a":
            return (True, True)
        if choice == "y":
            return (True, False)
        return (False, False)

    def _continue_shim(self, summary: str) -> bool:
        """Sync continue-prompt — blocks the background thread until the WS client responds.

        Mirrors ``_approval_shim``: emits a ``continue_prompt`` card to the client
        and blocks the agent worker thread (not the WS event loop) until a
        ``continue_response`` arrives. A timeout or a non-"y" choice stops the run.
        """
        req_id = str(uuid.uuid4())
        self.out_q.put({
            "type": "continue_prompt",
            "id": req_id,
            "summary": summary,
        })
        # No timeout: keep the agent parked until answered (Stop cancels).
        response = self._await_response(self._continue_q)
        if response is None:
            return False
        return response.get("choice", "n") == "y"

    def _question_shim(self, questions: list) -> dict:
        """Sync clarification questions — blocks the worker thread until answered.

        Mirrors ``_continue_shim``: emits a ``user_question`` card carrying the whole
        batch of questions to the client and blocks the agent worker thread (not the
        WS event loop) until a ``user_question_response`` arrives. The frontend shows
        the questions one at a time and returns all ``answers`` together. A cancel
        returns no answers so the agent proceeds with its best judgment.
        """
        req_id = str(uuid.uuid4())
        self.out_q.put({
            "type": "user_question",
            "id": req_id,
            "questions": list(questions),
        })
        # No timeout: keep the agent parked until answered (Stop cancels).
        response = self._await_response(self._question_q)
        if response is None:
            return {"answers": []}
        answers: list[dict] = []
        for a in response.get("answers") or []:
            a = a or {}
            answers.append({
                "selected": [str(s) for s in (a.get("selected") or [])],
                "other_text": a.get("otherText") or a.get("other_text") or None,
            })
        return {"answers": answers}

    # ── Public API ────────────────────────────────────────────────────────────

    def full_history(self) -> list | None:
        """The structured transcript from the last completed query (system excluded).

        Mirrors what the CLI chat loop uses in full-context mode: the full message
        list ``[*prior_history, user_turn, assistant(tool_calls), tool_results,
        final_assistant]`` with the model's chain-of-thought already stripped (see
        ``_process_response`` in the agent loop). Returns ``None`` when no query has
        completed yet so callers can fall back to flattened history.
        """
        if self._agent is None:
            return None
        msgs = getattr(self._agent, "_last_full_messages", None)
        return list(msgs) if msgs else None

    def live_history(self) -> list | None:
        """The in-flight transcript of the query currently running (system excluded).

        The agent mutates this list in place at every step, so reading it gives the
        context as it grows *during* a turn — which is what the context bar needs.
        ``None`` when no query is running.
        """
        if self._agent is None:
            return None
        msgs = getattr(self._agent, "_live_messages", None)
        if not msgs:
            return None
        return list(msgs)[1:]  # drop the system message (counted as overhead)

    def export_agent_state(self) -> dict:
        """Export carry_context from the agent."""
        if self._agent is not None:
            return self._agent.export_state()
        return {}

    def load_agent_state(self, state: dict) -> None:
        """Restore agent carry_context."""
        if self._agent is not None:
            self._agent.load_state(state)

    def submit_query(self, text: str, history: list) -> None:
        self._query_q.put({"text": text, "history": history})
        self._query_event.set()  # wake the query loop immediately

    def submit_steer(self, text: str) -> None:
        """Queue a message to inject into the RUNNING agent at its next step boundary."""
        self._steer_q.put(text)

    def is_busy(self) -> bool:
        """True while a query task is in flight (used to route steer vs. new query)."""
        return self._current_task is not None

    def reset_session_guards(self) -> None:
        """Clear session-scoped guard state on session change.

        Out-of-workspace approvals are session-scoped: a new session starts with a
        clean slate.
        """
        if self._agent is not None:
            try:
                self._agent.approvals.reset_allowed_paths()
            except Exception:
                pass

    def flush_prompts(self) -> None:
        """Drop every pending human-pause answer and steer message.

        Called right after ``cancel()`` when the turn they belong to is abandoned: a
        queued answer left behind would be handed to the *next* turn's prompt,
        approving something the user never saw.
        """
        for q in (self._approval_q, self._continue_q, self._question_q):
            while True:
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break
        self._drain_steer_q()

    def _drain_steer_q(self) -> list[str]:
        """Pop and return all queued steer messages (the agent's ``_poll_steer``)."""
        out: list[str] = []
        while True:
            try:
                out.append(self._steer_q.get_nowait())
            except _queue.Empty:
                break
        return out

    def _register_bg_job(self, descriptor: dict) -> bool:
        """Register a completion watcher for a detached run (the agent's hook).

        Called on the worker loop from the agent's tool dispatch. Dedups on
        ``job_key`` so re-launching the same job never spawns a second watcher.
        Returns True when a watcher is (already) active — the loop uses this to
        tell the model it may end its turn.
        """
        if not isinstance(descriptor, dict):
            return False
        job_key = str(descriptor.get("job_key") or descriptor.get("run_dir") or "")
        if not job_key:
            return False
        existing = self._bg_jobs.get(job_key)
        if existing is not None and not existing.done():
            return True  # already watched
        try:
            task = asyncio.get_event_loop().create_task(
                self._watch_job(job_key, descriptor))
        except RuntimeError:
            return False
        self._bg_jobs[job_key] = task
        return True

    async def _watch_job(self, job_key: str, descriptor: dict) -> None:
        """Poll a detached run to completion, then emit ``job_complete`` on ``out_q``.

        An independent task on the worker loop that outlives the launching turn.
        Read-only status polling only (safe to interleave with an active turn on the
        single-threaded loop). The WS session handles the UI notification and the
        auto-resume, because it owns the conversation history (see ``_drain_loop``).
        """
        status_op  = descriptor.get("status_op") or {}
        summary_op = descriptor.get("summary_op") or {}
        status_tool = status_op.get("tool")
        if not status_tool:
            self._bg_jobs.pop(job_key, None)
            return

        interval, max_interval = 5.0, 30.0
        terminal = {"done", "crashed", "unknown"}
        state = "running"
        try:
            while True:
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)
                try:
                    raw = await self._agent._run_tool(
                        status_tool, dict(status_op.get("args") or {}))
                    payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
                    state = str(payload.get("state") or "")
                except asyncio.CancelledError:
                    raise
                except Exception:
                    continue  # transient poll failure — retry next tick
                if state in terminal:
                    break
        except asyncio.CancelledError:
            self._bg_jobs.pop(job_key, None)
            return

        summary: dict = {}
        summary_tool = summary_op.get("tool")
        if summary_tool:
            try:
                raw = await self._agent._run_tool(
                    summary_tool, dict(summary_op.get("args") or {}))
                summary = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                summary = {}

        self.out_q.put({
            "type":    "job_complete",
            "job_key": job_key,
            "server":  descriptor.get("server"),
            "kind":    descriptor.get("kind"),
            "state":   state,
            "summary": summary,
        })
        self._bg_jobs.pop(job_key, None)

    def resolve_approval(self, choice: str, approved_files: list | None = None) -> None:
        self._approval_q.put({"choice": choice, "approved_files": approved_files})

    def resolve_continue(self, choice: str) -> None:
        self._continue_q.put({"choice": choice})

    def resolve_question(self, answers: list | None) -> None:
        self._question_q.put({"answers": answers or []})

    def set_mode(self, mode: str) -> str:
        """Apply *mode*, returning "" on success or the reason it was rejected.

        The caller reports the failure to the front-end — silently swallowing it
        left the webview showing a mode the agent was never switched to.
        """
        if self._agent is None:
            return "Agent is not ready yet."
        try:
            self._agent.set_mode(mode)
        except ValueError as exc:
            return str(exc)
        return ""

    def set_batch(self, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_batch_mode(enabled)

    def set_thinking_depth(self, level: int) -> None:
        if self._agent is not None:
            self._agent.set_thinking_depth(level)

    def set_thinking(self, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_thinking(enabled)

    def set_thinking_budget(self, budget: int) -> None:
        if self._agent is not None:
            self._agent.set_thinking_budget(budget)

    def set_streaming(self, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_streaming(enabled)

    def set_context_mode(self, mode: str) -> None:
        if self._agent is not None:
            try:
                self._agent.set_context_mode(mode)
            except ValueError:
                pass

    def set_enforcement(self, level: str) -> None:
        if self._agent is not None:
            try:
                self._agent.set_enforcement(level)
            except ValueError:
                pass

    def get_context_mode(self) -> str:
        if self._agent is not None:
            return getattr(self._agent, "context_mode", "compact")
        return "compact"

    def get_enforcement(self) -> str:
        if self._agent is not None:
            return getattr(self._agent, "enforcement", "strict")
        return "strict"

    def toggles_state(self) -> dict:
        if self._agent is not None:
            return self._agent.toggles_state()
        return {"servers": [], "skills": [], "nudges": []}

    def set_server_enabled(self, name: str, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_server_enabled(name, enabled)

    def set_skill_enabled(self, name: str, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_skill_enabled(name, enabled)

    def set_nudge_enabled(self, name: str, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_nudge_enabled(name, enabled)

    def resources_snapshot(self) -> list[dict]:
        """List attachable MCP resources for the frontend picker.

        The registry (``agent.resources``) is populated once at connect and never
        mutated afterward, so this read-only snapshot is safe to take directly from
        the WS event loop without hopping to the worker loop.
        """
        if self._agent is None:
            return []
        out: list[dict] = []
        for uri, info in sorted((self._agent.resources or {}).items()):
            out.append({
                "uri": uri,
                "name": info.get("name", uri),
                "description": info.get("description", ""),
                "mimeType": info.get("mimeType"),
            })
        return out

    def resolve_resources(self, text: str) -> Any:
        """Resolve @-mentions by reading resources on the worker's OWN loop.

        The MCP ``ClientSession`` objects are bound to this worker's event loop, so the
        read must run there — we schedule it via ``run_coroutine_threadsafe`` and return
        a ``concurrent.futures.Future`` the caller awaits with ``asyncio.wrap_future``.
        """
        if self._agent is None or self._loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result((text, []))
            return fut
        return asyncio.run_coroutine_threadsafe(
            augment_query_with_resources(self._agent, text), self._loop
        )

    def _count_tokens(self, text: str) -> int:
        """Token count for *text* — exact when the backend has a tokenizer.

        ``allow_network=False``: this runs inline on the streaming path, which must
        not stall on a tokenize round-trip mid-answer; the cached-or-heuristic value
        is close enough for a size readout. Never raises.
        """
        if not text:
            return 0
        try:
            from ...query_engine.backends.factory import get_backend
            return get_backend().count_text_tokens(self.model, text, allow_network=False)
        except Exception:
            return 0

    def context_overhead_tokens(self) -> int:
        """Fixed prompt overhead (tokens) sent on *every* LLM call besides history.

        This is the base system prompt plus the full tools schema — together
        ~8–12k tokens on this deployment. The context bar must include it,
        otherwise the displayed usage hides the very tokens that overflow the
        window. Best-effort, cached via the backend's token cache; never raises.
        """
        if self._agent is None:
            return 0
        try:
            from ...query_engine.backends.factory import get_backend
            from ...agent_core import build_base_system_content
            backend = get_backend()
            total = backend.count_text_tokens(
                self.model, build_base_system_content(), allow_network=False
            )
            tools = getattr(self._agent, "tools", None)
            if tools:
                total += backend.count_text_tokens(
                    self.model, json.dumps(tools), allow_network=False
                )
            return total
        except Exception:
            return 0

    def cancel(self) -> bool:
        """Cancel the currently running query task. Returns True if a task was cancelled."""
        # Set the cancel flag first so _stream_chat aborts mid-chunk immediately,
        # without waiting for the next asyncio await point.
        if self._agent is not None:
            self._agent._cancel_flag.set()
        task = self._current_task
        if task is not None and not task.done() and self._loop is not None:
            self._loop.call_soon_threadsafe(task.cancel)
            return True
        return False

    def drain(self) -> list[dict]:
        """Drain all pending output events (non-blocking), stamped with their session.

        Single choke point for everything the engine emits, so the stamp is applied
        here rather than at the dozens of emit sites. Events produced outside a query
        carry ``None`` and are never filtered.
        """
        events: list[dict] = []
        while True:
            try:
                ev = self.out_q.get_nowait()
            except _queue.Empty:
                break
            if isinstance(ev, dict):
                ev.setdefault("session_id", self._query_session_id)
            events.append(ev)
        return events

    def shutdown(self) -> None:
        # Cancel any in-flight background-job watchers on the worker loop.
        for task in list(self._bg_jobs.values()):
            if self._loop is not None and not task.done():
                self._loop.call_soon_threadsafe(task.cancel)
        self._query_q.put(None)
        self._query_event.set()

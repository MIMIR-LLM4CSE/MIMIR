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
    _ORIGINAL_STDOUT,
    _ROUTER,
    _todo_file_for_session,
    augment_query_with_resources,
)

import asyncio
import concurrent.futures
import json
import logging
import os
import queue as _queue
import threading
import uuid
from typing import Any, NamedTuple

from ... import human_pause
from ...config.constants import DEFAULT_SUBAGENT_LEVEL
from ...tool_execution.formatter import parse_tool_payload

logger = logging.getLogger(__name__)


def _direct_opener():
    """An opener that ignores HTTP_PROXY/HTTPS_PROXY.

    The LLM backend is an internal cluster address; a corporate proxy has no
    route to it, so a proxied probe hangs until its own timeout instead of
    failing fast. ``urlopen`` consults the environment by default, unlike the
    ``trust_env=False`` httpx clients the backends use — same posture, stated
    explicitly here.
    """
    import urllib.request as _r
    return _r.build_opener(_r.ProxyHandler({}))


# Consecutive status probes that come back without a state we recognise before a
# watcher gives up. A probe that stopped working — a policy violation, a dead server,
# a renamed op — is indistinguishable from a running job if we only look for a
# terminal state, and a watcher that keeps polling one leaves the agent waiting on a
# wake that can never come. Five ticks is past the exponential backoff's early, short
# intervals, so a single transient failure never trips it.
_UNREADABLE_POLL_LIMIT = 5


def _labelled_questions(questions: list, prefix: str) -> list:
    """Mark each question with *prefix* (empty prefix: the list, untouched)."""
    if not prefix:
        return questions
    out = []
    for q in questions:
        if isinstance(q, dict) and q.get("question"):
            q = {**q, "question": f"{prefix}{q['question']}"}
        out.append(q)
    return out


class _Watch(NamedTuple):
    """A live background-job watcher: the polling task, and what it is watching.

    The descriptor is kept next to the task because the dispatch guard compares an
    incoming call against the *job's own* ``status_op`` — registry data travelling on
    the descriptor, the same reason the watcher itself can poll generically. Holding
    only the task would have forced the guard to name a tool.
    """
    task: asyncio.Task
    descriptor: dict


def _first_line(value: Any) -> str:
    """The first line of an error payload, for a one-line reason. "" when there is none."""
    text = str(value or "").strip()
    return text.splitlines()[0][:200] if text else ""


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
        self._question_q: _queue.Queue[dict] = _queue.Queue()  # WS → question shim
        self._query_q: _queue.Queue[dict | None] = _queue.Queue()  # WS → query loop
        self._steer_q: _queue.Queue[str] = _queue.Queue()  # WS → running agent (mid-run steering)
        # The card a parked turn is waiting on, set for exactly as long as it waits.
        # Read by a connection that arrives while the wait is on (see _emit_prompt).
        self._pending_prompt: dict | None = None
        # Event signalled whenever a new item is placed on _query_q so the
        # background loop wakes up immediately instead of waiting out the poll interval.
        self._query_event = threading.Event()

        self._current_task: asyncio.Task | None = None
        self.active_session_id: str | None = None  # kept in sync by _Session
        # Session a running query belongs to, captured when it starts. One worker
        # serves every session, so events must carry the session they were produced
        # for or a turn that outlives a switch lands in the wrong conversation.
        self._query_session_id: str | None = None
        # Background-job watchers: job_key -> _Watch(task, descriptor), polling a
        # detached run to completion. Registered by the agent loop via _register_bg_job
        # (below) and read back by _watched_bg_jobs, which the dispatch guard uses.
        self._bg_jobs: dict[str, _Watch] = {}
        # Set by the front-end when the user leaves a conversation whose turn is parked
        # on them: every wait of that turn returns at once instead (query_engine.deferral).
        self._defer = threading.Event()
        # The answer a resume turn carries, handed to the first prompt of its kind
        # instead of putting the card up again: ``{"type", "response"}``.
        self._preanswer: dict | None = None
        # The raw questions of the pending question card — how a deferred question is
        # matched to the call that asked it.
        self._pending_questions: list | None = None

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
        """Poll the LLM backend until it answers, or fail fast when it never will.

        The request is the one the agent actually depends on — the model list for
        every OpenAI-compatible endpoint (``/v1/models``), ``/api/tags`` for Ollama —
        asked with the same client, proxy posture and TLS policy the backends
        themselves use. A probe that asks a different question of a different client
        than the chat path can fail where the chat path succeeds, which is exactly
        what a readiness check must never do.

        ``/health`` is kept as a fallback for a local vLLM: it answers while the
        engine is still loading weights, before ``/v1/models`` does. It is only a
        fallback because it lives at the server root, which an ingress route in front
        of the endpoint usually does not expose.

        Not every failure is worth waiting on. A refused certificate or a route that
        is not there answers identically on the first attempt and on the hundredth,
        so those stop the wait immediately, naming the URL and the reason, instead of
        spending the full timeout on a verdict already known.

        Progress goes to the client *and* to stdout: during startup the WS server has
        not bound its port yet, so the client that would show the spinner does not
        exist, and an attempt reported only there is an attempt reported to no one.

        Controlled by env vars:
        - MIMIR_BACKEND_TIMEOUT  (seconds, default 600)
        - MIMIR_BACKEND_POLL_INTERVAL (seconds, default 5)
        """
        import httpx

        from ....servers._shared.embed import verify_ssl

        try:
            from ...config.models import LLM_BACKEND, RAY_BASE_URL, VLLM_BASE_URL
        except ImportError:
            LLM_BACKEND = os.environ.get("LLM_BACKEND", "vllm")
            VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
            RAY_BASE_URL = os.environ.get("RAY_BASE_URL", "http://127.0.0.1:8000")

        # Env first, like the base URLs below: ``--backend`` writes LLM_BACKEND into
        # the environment after this module was imported, so the constant captured at
        # import time is the *shell's* backend, not the one this server was launched
        # for. Reading it would probe one endpoint while the agent talks to another.
        backend = os.environ.get("LLM_BACKEND", LLM_BACKEND)

        if backend in ("vllm", "ray"):
            env_var = "VLLM_BASE_URL" if backend == "vllm" else "RAY_BASE_URL"
            default = VLLM_BASE_URL if backend == "vllm" else RAY_BASE_URL
            base = os.environ.get(env_var, default).rstrip("/")
            health_url = base + ("/models" if base.endswith("/v1") else "/v1/models")
            # Only a local vLLM serves this; behind a route it 404s, which the
            # fallback treats as "no answer here" rather than as a failure.
            fallback_url = f"{base}/health" if backend == "vllm" else ""
        else:
            base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            health_url = f"{base}/api/tags"
            fallback_url = ""

        timeout = int(os.environ.get("MIMIR_BACKEND_TIMEOUT", "600"))
        poll = float(os.environ.get("MIMIR_BACKEND_POLL_INTERVAL", "5"))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_progress = loop.time()
        progress_interval = 10.0

        def _say(text: str) -> None:
            self.out_q.put({"type": "output", "text": text})
            print(text.rstrip("\n"), file=_ORIGINAL_STDOUT, flush=True)

        def _get(url: str) -> tuple[bool, str]:
            """(ready, permanent failure reason) for one GET. Empty reason: retry."""
            try:
                # trust_env=False: a corporate proxy has no route to the cluster and
                # swallows the request instead of refusing it. verify_ssl(): the same
                # switch the chat and embedding clients read, so the probe trusts
                # exactly what they trust.
                with httpx.Client(trust_env=False, timeout=4.0, verify=verify_ssl()) as client:
                    resp = client.get(url)
            except httpx.ConnectError as exc:
                # A refused certificate is the one connection failure that will not
                # resolve itself: waiting cannot add a CA to the trust store.
                if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSLCertVerificationError" in str(exc):
                    return False, (
                        f"the server at {url} answered, but its security certificate "
                        f"is not trusted (common for internal servers). If you trust "
                        f"this server, untick “Vllm Verify Ssl” in the MIMIR settings "
                        f"(or set VLLM_VERIFY_SSL=0), then retry. Details: {exc}"
                    )
                return False, ""
            except Exception:
                return False, ""
            if resp.status_code == 200:
                return True, ""
            # Up, but guarding the endpoint: the agent's own request carries the key.
            if resp.status_code in (401, 403):
                return True, ""
            if resp.status_code == 404:
                return False, f"404 — nothing is served at {url}"
            return False, ""

        def _check() -> tuple[bool, str]:
            ready, reason = _get(health_url)
            if ready or not fallback_url:
                return ready, reason
            # The engine may still be loading, when /health answers and /v1/models
            # does not. A fallback that answers outranks the primary's 404.
            alt_ready, _ = _get(fallback_url)
            return alt_ready, "" if alt_ready else reason

        _say(f"⏳ Waiting for LLM backend ({backend}) at {health_url} …\n")

        while True:
            ready, permanent = await loop.run_in_executor(None, _check)
            if ready:
                _say("✅ LLM backend is ready.\n")
                return
            if permanent:
                raise RuntimeError(f"LLM backend unreachable: {permanent}")

            now = loop.time()
            if now >= deadline:
                raise RuntimeError(
                    f"LLM backend did not become ready within {timeout} s "
                    f"(health URL: {health_url})"
                )
            if now - last_progress >= progress_interval:
                elapsed = int(now - (deadline - timeout))
                _say(f"⏳ Still waiting for LLM backend … ({elapsed}s elapsed)\n")
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
            # The twin of the hook above: what the dispatch guard reads to know a run
            # is already being watched, so asking whether it is done can be refused.
            agent._watched_background_jobs = self._watched_bg_jobs

            self._agent = agent
            self.out_q.put({"type": "ready", "model": self.model, "agent_ready": True})
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
        # /api/generate with an empty prompt is Ollama's "load this into VRAM" call.
        # It has no equivalent on the other backends, and firing it regardless meant
        # every vLLM/Ray session opened with a request to a localhost Ollama that
        # isn't there — a wasted connection, and a misleading one in a trace.
        if _os.environ.get("LLM_BACKEND", "vllm").lower() != "ollama":
            return
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
            with _direct_opener().open(req, timeout=120):
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
            self._query_session_id = item.get("session_id") or self.active_session_id
            await self._run_query(item)
            self._current_task = None
            self._query_session_id = None

    async def _run_query(self, item: dict) -> None:
        query = item.get("text", "")
        history = item.get("history", [])
        resume = item.get("resume")
        mode = None
        if resume is not None:
            mode = resume.get("mode")
            self._preanswer = {"type": resume.get("prompt", {}).get("type"),
                               "response": item.get("answer") or {}}
        if self._agent is not None:
            self._agent._deferred_prompts = []
            self._agent._deferred_turn = None

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
                    mode=mode,
                    resume=resume,
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
            # Deferral is scoped to the turn it was asked for, like the cancel flag.
            self._defer.clear()
            self._preanswer = None

        answer_ev: dict = {"type": "answer", "text": answer, "cancelled": cancelled,
                           # Stamped here: the turn is over by the time the drain
                           # loop reads this, and its session must not be guessed.
                           "session_id": self._query_session_id}
        # The turn's own transcript travels with its answer. Read later from the agent,
        # it may already be the next turn's: a queued turn starts, and resets it, as
        # soon as this one ends. Private keys — the session strips them before sending.
        if self._agent is not None:
            full = getattr(self._agent, "_last_full_messages", None)
            answer_ev["_full"] = list(full) if full else None
            answer_ev["_turn_start"] = getattr(self._agent, "_last_turn_start", None)
        deferred = self._deferred_record()
        if deferred is not None and not cancelled:
            answer_ev["_deferred"] = deferred

        # Push current todo state.
        self._push_todos()
        self.out_q.put(answer_ev)
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

    def _deferred_record(self) -> dict | None:
        """What the turn that just ended was set aside on, or None.

        The loop's own record (which calls, which mode) plus the card to put back:
        the first prompt deferred, whose id the answer will come back with.
        """
        agent = self._agent
        turn = getattr(agent, "_deferred_turn", None) if agent is not None else None
        prompts = getattr(agent, "_deferred_prompts", None) if agent is not None else None
        if not turn or not prompts:
            return None
        return {**turn, "prompt": prompts[0]["prompt"]}

    def defer(self) -> bool:
        """Set the parked turn aside instead of cancelling it (see deferral).

        True when a turn was parked on a prompt and is now unwinding; False when there
        was nothing to set aside, and the caller should fall back to cancelling.
        """
        if self._pending_prompt is None or not self.is_busy():
            return False
        self._defer.set()
        return True

    def is_parked(self) -> bool:
        """True while the running turn waits on the user."""
        return self.is_busy() and self._pending_prompt is not None

    def submit_resume(self, record: dict, answer: dict, history: list,
                      session_id: str | None) -> None:
        """Queue the turn that picks a deferred one up with the user's *answer*."""
        self._query_q.put({"text": record.get("query", ""), "history": history,
                           "session_id": session_id, "resume": record,
                           "answer": answer})
        self._query_event.set()

    def _emit_prompt(self, payload: dict, questions: list | None = None) -> None:
        """Send a card the turn is about to park on, and remember it while it waits.

        One worker serves every connection, and it outlives them: a socket that drops
        while the agent is parked leaves the card on a client that no longer exists,
        and the turn waiting on an answer nobody can give any more. Every later query
        queues behind that wait — the query loop is serial — so the session reads as
        hung, with nothing on screen to explain it. Kept here, the card can be put
        back in front of whoever reconnects (``_Session._resend_parked_prompt``).
        """
        self._pending_prompt = dict(payload)
        self._pending_questions = questions
        # Nobody is there to read it (deferring), or the answer is already in hand
        # (resuming): the wait below settles it without a card.
        if self._deferring() or self._preanswer_for(payload) is not None:
            return
        self.out_q.put(payload)

    def _deferring(self) -> bool:
        defer = getattr(self, "_defer", None)
        return defer is not None and defer.is_set()

    def _preanswer_for(self, payload: dict | None) -> dict | None:
        pre = getattr(self, "_preanswer", None)
        if pre is None or payload is None or pre.get("type") != payload.get("type"):
            return None
        return pre

    def pending_prompt(self) -> dict | None:
        """The card the parked turn is waiting on, once it is nobody's to deliver.

        None while the card is still queued: the drain loop will hand it to whoever is
        connected, and returning it here as well would put the same card on screen
        twice. That is not merely cosmetic — an approval renders as one card carrying
        both ids and is then answered twice, leaving a spare answer on the queue for
        the *next* prompt to consume, which is the exact failure ``flush_prompts``
        exists to prevent.
        """
        prompt = self._pending_prompt
        if prompt is None:
            return None
        with self.out_q.mutex:
            queued = [ev.get("id") for ev in self.out_q.queue if isinstance(ev, dict)]
        return None if prompt.get("id") in queued else prompt

    def _await_response(self, q: "_queue.Queue[dict]") -> dict | None:
        """Block until a WS response lands on ``q`` — with no wall-clock timeout.

        An unanswered approval/question must keep the agent *parked*: it
        must never silently proceed just because the user was slow to respond.
        So we wait indefinitely instead of timing out. To stay responsive to the
        Stop button, we poll in short slices and bail the moment the agent's
        cancel flag is set (from the WS thread), returning ``None`` for cancelled.

        This is the single seam every WS prompt (approval, out-of-workspace path,
        question) blocks on, so it is where the wait is marked as *human*
        time — excluded from the tool-call timeout budget it sits inside.
        """
        pre = self._preanswer_for(getattr(self, "_pending_prompt", None))
        if pre is not None:
            # One answer, one prompt: a second card in the resumed step is asked live.
            self._preanswer = None
            self._pending_prompt = None
            return pre.get("response") or {}
        try:
            with human_pause.human_pause():
                while True:
                    agent = self._agent
                    if agent is not None and agent._cancel_flag.is_set():
                        return None
                    if self._deferring():
                        self._record_deferral()
                        return None
                    try:
                        return q.get(timeout=0.25)
                    except _queue.Empty:
                        continue
        finally:
            # Answered, cancelled or raised through: the turn is not parked any more,
            # and a card resent past this point would be one nothing is waiting on.
            self._pending_prompt = None
            self._pending_questions = None

    def _record_deferral(self) -> None:
        """Note the prompt being set aside, and which call it holds up."""
        from ...query_engine.deferral import CURRENT_CALL_ID

        agent = self._agent
        if agent is None or self._pending_prompt is None:
            return
        agent._deferred_prompts = [*(getattr(agent, "_deferred_prompts", None) or []), {
            "call_id": CURRENT_CALL_ID.get(),
            "prompt": dict(self._pending_prompt),
            "questions": list(self._pending_questions or []),
        }]

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
            "label": f"{self._detached_prefix()}{label}" if label else label,
        }
        self._emit_prompt(payload)

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
        self, paths: list[str], tool_name: str, arguments: dict | None = None
    ) -> tuple[bool, bool]:
        """Out-of-workspace access prompt — reuses the approval UI (y/n/a).

        Carries the same descriptive payload as a normal approval (canonical label,
        the call's real arguments, the owning server) so the card reads like any
        other tool card; ``oow_paths`` is what makes it an out-of-workspace prompt and
        names the offending paths. Returns (approved, always). Blocks the worker thread
        on the approval queue like ``_approval_shim``; the engine's out-of-workspace
        gate records the grants.

        **One card for the whole call.** Every outside path the call names travels in a
        single payload: the user judges the command, not each of its operands, and the
        per-path loop this replaced parked the agent on one queue behind several
        identical questions.
        """
        from ...context.capabilities import IRREVERSIBLE, label_for
        from ...tool_execution.tool_status_messages import shorten_display_args

        agent = self._agent
        arguments = arguments or {}
        scope = (f"this path ({os.path.basename(paths[0])})" if len(paths) == 1
                 else f"these {len(paths)} paths")
        req_id = str(uuid.uuid4())
        self._emit_prompt({
            "type": "approval",
            "id": req_id,
            "tool": tool_name,
            "server": agent.tool_owner.get(tool_name, "filesystem"),
            "args": arguments or {"path": paths[0]},
            "risk": agent.approvals.describe_risk(tool_name),
            # Reaching outside the workspace is irreversible by situation, not by tool:
            # whatever the tool's own level, nothing here can undo a write landing
            # outside the sandbox, so the card must not soften it to the tool's rating.
            "reversibility": IRREVERSIBLE,
            "scope": scope,
            # Short label in the header; the card renders `oow_paths` beneath it as an
            # explicit "outside workspace" line. The absolute paths are the decision
            # being made, so they are never the thing that gets shortened.
            "label": label_for(
                tool_name,
                shorten_display_args(tool_name, arguments or {}, agent.tool_caps),
                agent.tool_caps,
            ),
            "oow_paths": list(paths),
            # Kept alongside the list so a client built against the single-path payload
            # still renders a path rather than nothing.
            "oow_path": paths[0],
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

    def _question_shim(self, questions: list) -> dict:
        """Sync clarification questions — blocks the worker thread until answered.

        Mirrors ``_approval_shim``: emits a ``user_question`` card carrying the whole
        batch of questions to the client and blocks the agent worker thread (not the
        WS event loop) until a ``user_question_response`` arrives. The frontend shows
        the questions one at a time and returns all ``answers`` together. A cancel
        returns no answers so the agent proceeds with its best judgment.
        """
        req_id = str(uuid.uuid4())
        self._emit_prompt({
            "type": "user_question",
            "id": req_id,
            "questions": _labelled_questions(list(questions), self._detached_prefix()),
        }, questions=list(questions))
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

    def last_turn_start(self) -> int | None:
        """Index in :meth:`full_history` where the last completed turn's own messages begin.

        The loop's own answer to "which of these did this turn produce", so a caller
        keeping a record of its own need not infer it from the length it submitted —
        an inference the in-turn budget rewrites invalidate. ``None`` when the boundary
        cannot be given honestly and the caller must fall back.
        """
        if self._agent is None:
            return None
        start = getattr(self._agent, "_last_turn_start", None)
        return start if isinstance(start, int) else None

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

    def submit_query(self, text: str, history: list,
                     session_id: str | None = None) -> None:
        """Queue a turn. ``session_id`` names the conversation it belongs to.

        Omitted, the turn belongs to whichever session is active when it starts —
        the ordinary case, a user typing. A background-job wake passes it explicitly,
        because the conversation that launched the job may no longer be on screen.
        """
        self._query_q.put({"text": text, "history": history, "session_id": session_id})
        self._query_event.set()  # wake the query loop immediately

    def submit_steer(self, text: str) -> None:
        """Queue a message to inject into the RUNNING agent at its next step boundary."""
        self._steer_q.put(text)

    def agent_ready(self) -> bool:
        """Whether the agent exists yet.

        The socket is bound long before this is true: the worker waits on the LLM
        backend first, which on a cold vLLM is minutes. The same test ``set_model``
        applies, and the honest one — ``_ready`` is set in a ``finally`` and is
        therefore also set when construction raised.
        """
        return self._agent is not None

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
        self._pending_prompt = None
        for q in (self._approval_q, self._question_q):
            while True:
                try:
                    q.get_nowait()
                except _queue.Empty:
                    break
        self._drain_steer_q()

    def _detached_prefix(self) -> str:
        """A marker for a prompt raised by a turn the user is not currently reading.

        A background-job wake resumes the session that launched the job, which may not
        be the one on screen. Its approval and question cards still have to be shown —
        the turn is parked until they are answered — so they say which conversation
        they belong to instead of appearing to come from the one being read.
        """
        running = self._query_session_id
        if running and running != self.active_session_id:
            return "⏱ background session · "
        return ""

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

        The session that launched the job is captured here and travels with the
        watcher: a two-hour build outlives the conversation on screen, and the wake
        belongs to the conversation that asked for it, not to whichever one the user
        happens to be reading when it lands.
        """
        if not isinstance(descriptor, dict):
            logger.warning("background-job registration refused: descriptor is %s, "
                           "not a dict", type(descriptor).__name__)
            return False
        job_key = str(descriptor.get("job_key") or descriptor.get("run_dir") or "")
        if not job_key:
            logger.warning("background-job registration refused: descriptor carries "
                           "neither 'job_key' nor 'run_dir' (keys: %s)",
                           sorted(descriptor))
            return False
        existing = self._bg_jobs.get(job_key)
        if existing is not None and not existing.task.done():
            return True  # already watched
        session_id = self._query_session_id or self.active_session_id
        try:
            # get_running_loop, not get_event_loop: the latter can hand back a loop
            # that is not running, and a task created on one of those never polls
            # anything while registration still reports success. That combination is
            # the worst of both — the model is promised a resume, and nothing is
            # holding the run. Registering is only ever done from the worker loop, so
            # requiring a running one states the real precondition.
            task = asyncio.get_running_loop().create_task(
                self._watch_job(job_key, descriptor, session_id))
        except RuntimeError:
            # No running loop to host the watcher. Logged rather than swallowed: this
            # is the point where a launched run stops being tracked by anything, and
            # every minute after it is a run finishing into silence.
            logger.warning("background-job registration failed for %r: no running "
                           "event loop to host the watcher", job_key, exc_info=True)
            return False
        self._bg_jobs[job_key] = _Watch(task, descriptor)
        return True

    def _watched_bg_jobs(self) -> list[dict]:
        """Descriptors of the runs a watcher is currently holding (the agent's hook).

        The deterministic half of the dispatch guard: a job is in this list or it is
        not, so nothing has to be inferred about what the model meant. Read on the
        worker loop, which is also the only thread that writes ``_bg_jobs``.

        Finished tasks are filtered rather than trusted to have been popped: a watcher
        that raised leaves its entry behind, and a guard that refused calls on a job
        nobody is watching would block the one question the model still needs to ask.
        """
        return [w.descriptor for w in self._bg_jobs.values()
                if isinstance(w.descriptor, dict) and not w.task.done()]

    async def _watch_job(self, job_key: str, descriptor: dict,
                         session_id: str | None = None) -> None:
        """Poll a detached run to completion, then emit ``job_complete`` on ``out_q``.

        An independent task on the worker loop that outlives the launching turn.
        Read-only status polling only (safe to interleave with an active turn on the
        single-threaded loop). The WS session handles the UI notification and the
        auto-resume, because it owns the conversation history (see ``_drain_loop``).

        The probes run with observations off: a watcher tick is this worker asking a
        question, not a step the model took, and recording it as one would credit the
        agent with work it did not do.

        The event carries the descriptor's own ops. The session that reads it knows
        nothing about what kind of job this was, so what it can say to the model comes
        from the descriptor and the summary rather than from anything it names itself.
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
        reason = ""
        unreadable = 0
        # Last (phase, percent) announced, so a poll that learned nothing says nothing.
        last_reported: tuple = ("", None)
        try:
            while True:
                await asyncio.sleep(interval)
                interval = min(interval * 1.5, max_interval)
                try:
                    raw = await self._agent._run_tool(
                        status_tool, dict(status_op.get("args") or {}),
                        record_observations=False)
                    # parse_tool_payload, not json.loads: the result is an envelope
                    # followed by its text blocks and any appended annotation, and a
                    # bare load reads that as one broken document. Five of those in a
                    # row and the watcher gives the run up as unreadable.
                    payload = parse_tool_payload(raw) if isinstance(raw, str) else (raw or {})
                    state = str((payload or {}).get("state") or "")
                    if not state:
                        reason = (_first_line(payload.get("error"))
                                  or f"'{status_tool}' returned no state")
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    payload = {}
                    state, reason = "", f"'{status_tool}' raised {type(exc).__name__}"
                if state in terminal:
                    break
                # What the run is doing, for as long as it is doing it. The blocking
                # wait had its own channel into the run; once detached, this poll is
                # the only thing still asking, so it is the only thing that can say.
                # Shape-driven like everything else here: whatever the status op
                # chose to report, passed on without being understood.
                phase = str((payload or {}).get("phase") or "")
                percent = (payload or {}).get("percent")
                if not isinstance(percent, (int, float)):
                    percent = None
                # Starts at ("", None), so a run that never reports is never sent,
                # while one whose count disappears is — a retraction is news too.
                if (phase, percent) != last_reported:
                    last_reported = (phase, percent)
                    self.out_q.put({
                        "type":       "job_progress",
                        "job_key":    job_key,
                        "phase":      phase,
                        "percent":    percent,
                        "session_id": session_id,
                    })
                if state:
                    unreadable = 0   # 'running', or any state the descriptor's op owns
                    continue
                # Not a state we can act on. Retried a few times in case it is
                # transient, then reported as unknown — an honest "I lost track of it"
                # reaches the user, where another silent tick never would.
                unreadable += 1
                if unreadable >= _UNREADABLE_POLL_LIMIT:
                    state = "unknown"
                    break
        except asyncio.CancelledError:
            self._bg_jobs.pop(job_key, None)
            return

        summary: dict = {}
        summary_tool = summary_op.get("tool")
        if summary_tool:
            try:
                raw = await self._agent._run_tool(
                    summary_tool, dict(summary_op.get("args") or {}),
                    record_observations=False)
                summary = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                summary = {}

        self.out_q.put({
            "type":       "job_complete",
            "job_key":    job_key,
            "server":     descriptor.get("server"),
            "kind":       descriptor.get("kind"),
            "state":      state,
            "summary":    summary,
            "status_op":  status_op,
            "summary_op": summary_op,
            "session_id": session_id,
            "reason":     reason if state == "unknown" else "",
        })
        self._bg_jobs.pop(job_key, None)

    def resolve_approval(self, choice: str, approved_files: list | None = None) -> None:
        self._approval_q.put({"choice": choice, "approved_files": approved_files})

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

    def set_model(self, model: str) -> str:
        """Switch the served model, returning "" on success or the reason it failed.

        Also updates ``self.model`` so ``ready``/profile reads report the new model.
        """
        if self._agent is None:
            return "Agent is not ready yet."
        try:
            self._agent.set_model(model)
        except ValueError as exc:
            return str(exc)
        self.model = model
        return ""

    def served_models(self) -> list[str]:
        """Model ids the endpoint is serving, or [] if it cannot say.

        The agent process is the one authority on this: it holds the address and the
        API key, and it reaches the cluster with the proxy posture the whole client
        uses. The VS Code panel used to learn the list only from its own probe in the
        extension host, so a probe that a corporate proxy swallowed left the user
        connected to a working endpoint with no way to switch model — the list was
        missing, not the capability. Reporting it from here makes the panel's picker
        independent of whether that probe got through.

        Never raises: a backend that cannot enumerate itself, or an endpoint that
        does not answer, reads as "nothing to offer" and the caller falls back to
        naming the active model.
        """
        if self._agent is None:
            return []
        try:
            from ...query_engine.backends.factory import get_backend
            return [m for m in get_backend().served_models() if isinstance(m, str) and m.strip()]
        except Exception:
            return []

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

    def set_subagent_level(self, level: str) -> str:
        """Set how far sub-agents may go, and return the rung in force."""
        if self._agent is None:
            return DEFAULT_SUBAGENT_LEVEL
        return self._agent.set_subagent_level(level)

    def get_subagent_level(self) -> str:
        return getattr(self._agent, "subagent_level", DEFAULT_SUBAGENT_LEVEL)

    def set_enforcement(self, level: str) -> None:
        if self._agent is not None:
            try:
                self._agent.set_enforcement(level)
            except ValueError:
                pass

    def set_temperature(self, value: float | None) -> None:
        if self._agent is not None:
            try:
                self._agent.set_temperature(value)
            except ValueError:
                pass

    def set_approval_mode(self, mode: str) -> None:
        """Switch who answers the approval cards — valid mid-run.

        Called from the WS event loop while the worker thread may be running a query,
        or parked on a card. It only rebinds an attribute the policy engine reads
        afresh at each tool call, so the new mode applies from the next call on with
        no queue involved. A card *already* on screen is answered by the client, which
        is the side that knows one is standing.
        """
        if self._agent is not None:
            try:
                self._agent.set_approval_mode(mode)
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

    def get_approval_mode(self) -> str:
        if self._agent is not None:
            return getattr(self._agent.approvals, "approval_mode", "manual")
        return "manual"

    def get_temperature_state(self) -> dict:
        """Whether this backend honours a temperature, and the one set for the model.

        ``value`` None is the model's own. Before the agent exists (the greeting is
        sent while the backend is still coming up) it is read from the stored
        preference, which is exactly what the agent will load.
        """
        from ...config import TEMPERATURE_BACKENDS
        backend = os.environ.get("LLM_BACKEND", "vllm").lower()
        if self._agent is not None:
            value = getattr(self._agent, "temperature", None)
        else:
            from ...config.preferences import load_temperature
            value = load_temperature(self.model)
        return {"supported": backend in TEMPERATURE_BACKENDS, "value": value}

    def get_thinking_profile(self) -> dict:
        """What the panel needs to draw a depth control this model can honour.

        The rung labels are the family's own (`low`/`high`/`max` for some, OpenAI's
        `low`/`medium`/`high` for others), and `can_disable` says whether an "off"
        rung would do anything — so the control never offers a setting the request
        builder cannot express.
        """
        backend = os.environ.get("LLM_BACKEND", "vllm").lower()
        if backend not in ("vllm", "ray"):
            # Ollama takes a `think` flag and Anthropic a thinking block; both are
            # plain on/off with a budget, i.e. the full ladder applies. Ray is not
            # here: its router drives vLLM engines, so the same per-family profiles
            # describe how those models are told to reason.
            return {"mechanism": "kwarg", "levels": [], "can_disable": True}
        from ...config.models import thinking_profile, thinking_can_disable
        profile = thinking_profile(self.model)
        return {
            "mechanism": profile["mechanism"],
            "levels": profile["levels"] if profile["mechanism"] == "effort" else [],
            "can_disable": thinking_can_disable(self.model),
        }

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

    def call_session_tool(self, tool: str, args: dict) -> Any:
        """Invoke *tool* on the worker's OWN loop; returns a Future of the decoded payload.

        Same constraint as :meth:`resolve_resources`: the MCP ClientSession objects are
        bound to this loop, so a tool call made from the WebSocket thread has to be
        scheduled onto it rather than awaited where it was asked for.

        This is what lets a slash command do housekeeping directly. These ops are
        reachable by the model too, but the person who wants to start an optimisation
        over — or to drop a memory that has gone stale and keeps being recalled into
        every prompt — should not have to ask the model to do it. Before this the only
        recourse was deleting files under a store whose path they had no reason to know.

        The call goes STRAIGHT to the owning MCP session, around the guardrail
        pipeline — the same bypass the CLI surface makes in
        ``chat_commands._call_platform_tool``, and for the same reason: approvals,
        plan-shape gates and the write policy exist to judge what the *model* asked
        for. Routed through ``_run_tool`` instead, a command the user typed was weighed
        as if the model had proposed it — in plan or ask mode the plan gate answered
        with a refusal string, so the command did nothing and said nothing — and every
        one of them scattered tool cards through the transcript on its way.
        """
        if self._agent is None or self._loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result({"status": "error", "error": "No agent session is running."})
            return fut

        agent = self._agent
        owner = (getattr(agent, "tool_owner", None) or {}).get(tool)
        if owner is None:
            fut = concurrent.futures.Future()
            fut.set_result({"status": "error",
                            "error": f"The server owning '{tool}' is not connected."})
            return fut

        async def _run() -> dict:
            try:
                raw = await agent.sessions[owner].call_tool(tool, dict(args))
                text = agent._normalize_tool_content(raw)
                return json.loads(text) if isinstance(text, str) else (text or {})
            except Exception as exc:
                return {"status": "error", "error": str(exc)}

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def panel_sections(self) -> Any:
        """Collect every section the connected servers can fill. Future of a list.

        Asked by capability, never by name: a server declares ``PANEL_REPORT`` and the
        section it fills, and a plugin server's section appears with no client edit.
        Each answer is whatever that server chose to show — the panel renders lines, it
        does not interpret them.

        Straight to the owning session, like the slash commands above: this is the
        user opening a drawer, not the model taking a step, and routing it through the
        guardrail pipeline would scatter tool cards through their transcript.
        """
        from ...context.capabilities import panel_sections as _declared

        if self._agent is None or self._loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result([])
            return fut
        agent = self._agent
        declared = _declared(getattr(agent, "tool_caps", None))

        async def _run() -> list[dict]:
            out: list[dict] = []
            for tool, spec in declared:
                owner = (getattr(agent, "tool_owner", None) or {}).get(tool)
                if owner is None:
                    continue
                try:
                    raw = await agent.sessions[owner].call_tool(tool, dict(spec.get("args") or {}))
                    text = agent._normalize_tool_content(raw)
                    payload = json.loads(text) if isinstance(text, str) else (text or {})
                except Exception as exc:
                    # One server that cannot answer costs its own section, never the
                    # panel: the others have already said something worth showing.
                    logger.debug("panel section %r failed: %s", tool, exc)
                    continue
                lines = payload.get("lines") or []
                detail = payload.get("detail") or ""
                if not lines and not detail:
                    # A server with nothing to say gets no heading. A section that has
                    # something to report about having nothing — "no optimisation
                    # session here" — says it in `detail` and still appears; one that
                    # returns neither is a facility this machine does not have, and a
                    # bare title is a heading the user reads to learn there is nothing
                    # to read.
                    continue
                out.append({
                    "section": spec.get("section", tool),
                    "title": payload.get("title") or spec.get("section", tool),
                    "lines": lines,
                    "detail": detail,
                })
            return out

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def watched_runs(self) -> list[dict]:
        """What is running outside the current turn: detached jobs the watcher holds.

        Read off the descriptors the servers put there, so this says what kind of run
        it is without knowing what any of them do.
        """
        return [
            {"job_key": d.get("job_key", "?"), "kind": d.get("kind", ""),
             "server": d.get("server", "")}
            for d in self._watched_bg_jobs()
        ]

    def compact_middle(self, middle: list) -> Any:
        """Summarize *middle* into one message, off the WS event loop.

        The session's pre-query budget check used to front-trim only — the oldest
        turns were dropped and what they established was simply forgotten — because
        summarizing means an LLM call and that call must not run on the WebSocket
        event loop. It runs here instead: scheduled on this worker's loop and handed
        to an executor thread, since ``compact_messages`` blocks on the backend.

        Returns a ``concurrent.futures.Future`` resolving to the summary messages,
        or to *middle* unchanged when summarization failed (``compact_messages``
        swallows its own errors) — the caller treats that as "no compaction".
        """
        if self._agent is None or self._loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result(middle)
            return fut

        async def _run() -> list:
            return await asyncio.get_running_loop().run_in_executor(
                None, self._agent.compact_messages, middle
            )

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

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

    def context_overhead_is_measured(self) -> bool:
        """True once a server-reported prompt size has replaced the estimate.

        Surfaced to the bar's tooltip because the two numbers answer different
        questions: before the first answer the overhead is this client's reading of
        the prompt it is about to send, after it the server's reading of what it
        received. Saying which one is on screen turns a figure that quietly shifts
        after the first turn into one the user can account for.
        """
        try:
            from ...query_engine.backends.factory import get_backend
            probe = getattr(get_backend(), "measured_prompt_overhead", None)
            return callable(probe) and probe(self.model) is not None
        except Exception:
            return False

    def context_overhead_tokens(self) -> int:
        """Fixed prompt overhead (tokens) sent on *every* LLM call besides history.

        The system prompt plus the tools schema — ~30k tokens with every tool server
        on, most of it the tools. The context bar must include it, otherwise the
        displayed usage hides the very tokens that actually overflow the window.

        Both halves are measured the way the query builds them, not approximated.
        The prompt comes from ``build_system_content_now`` in the current mode, so
        the mode-gated sections, the memory index and the checklist are counted;
        measuring the base doctrine alone under-reported the prompt by everything
        the modes add. The tools come from ``advertised_tools_for_mode``, so a
        disabled server and a read-only mode remove from the bar what they remove
        from the call; the raw registry over-reported them. What stays out of reach
        is ``tools_for_context``, which prunes against the query and therefore does
        not exist before there is one — it only ever removes, so this is a ceiling
        on the tool half and the bar errs high, never low.

        Once a call has come back, none of this is estimated any more: the server
        reports what it actually charged for the prompt, and ``measured_prompt_overhead``
        hands back that figure minus the history it came with. That number replaces the
        estimate for every later call, which is what makes the bar exact rather than
        merely close — and it is why the estimate above only has to carry the session
        as far as its first answer.

        Best-effort, cached via the backend's token cache; never raises.
        """
        agent = self._agent
        if agent is None:
            return 0
        try:
            from ...query_engine.backends.factory import get_backend
            from ...agent_core import build_base_system_content
            backend = get_backend()
            # getattr, not a direct call: a backend stub without the calibration
            # should fall through to the estimate, not lose the overhead entirely
            # to the except below.
            probe = getattr(backend, "measured_prompt_overhead", None)
            measured = probe(self.model) if callable(probe) else None
            if measured is not None:
                return measured
            mode = getattr(agent, "mode", "") or ""
            build = getattr(agent, "build_system_content_now", None)
            prompt = build(mode) if callable(build) else build_base_system_content()
            total = backend.count_text_tokens(self.model, prompt, allow_network=False)
            narrow = getattr(agent, "advertised_tools_for_mode", None)
            tools = narrow(mode) if callable(narrow) else getattr(agent, "tools", None)
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
        for watch in list(self._bg_jobs.values()):
            if self._loop is not None and not watch.task.done():
                self._loop.call_soon_threadsafe(watch.task.cancel)
        self._query_q.put(None)
        self._query_event.set()

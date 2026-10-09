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
# After the runtime: it reads the workspace root that the bootstrap above pins.
from .job_scan import (
    establish_baseline,
    has_baseline,
    scan_session,
)

import asyncio
import concurrent.futures
import json
import logging
import os
import queue as _queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from itertools import chain, repeat
from typing import Any

from ... import human_pause
from ...tool_execution.formatter import parse_tool_payload

logger = logging.getLogger(__name__)

# What a bounded prompt wait returns when its wall passes with nothing answered. A
# unique instance compared with ``is``, never read as a response: ``None`` already
# means cancelled, and an expired card has to be taken back down rather than treated
# as a user who said no.
TIMED_OUT: dict = {"timed_out": True}


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

# How long one of a watcher's own tool calls may take before it is given up on.
#
# There is no deadline anywhere beneath this: ``session.call_tool`` takes none, and the
# MCP sessions are built without a read timeout, so a status op that stops answering —
# a wedged server, a scheduler that hangs, a queue command that never returns — blocks
# the watcher for ever. That is the one failure that genuinely loses a run: the task
# stays in ``_bg_jobs``, so the agent is never released and never woken, the bulletins
# keep repeating a status frozen at the moment it stopped, and nothing anywhere says
# the run is no longer being watched. A watcher that dies at least says so.
#
# Generous on purpose: a status op is a cheap read, and this is the line between "slow"
# and "never", not a budget to work inside. A tick that crosses it is counted as
# unreadable, which is a case this loop already has — ``_UNREADABLE_POLL_LIMIT`` of
# them in a row and the run is reported as unknown, with the reason — so a stuck status
# op ends as an honest "I lost track of it" instead of silence.
_TOOL_CALL_TIMEOUT = 60.0

# When a background check-in fires, as seconds from the one before it: T+30s, T+2.5min,
# T+12.5min, T+42.5min, and hourly from there for as long as the run lasts. A detached
# run says nothing between its launch and its end, so a two-hour build that went wrong
# in its third minute is found out in its hundred-and-twentieth.
#
# The ramp widens the way the watcher's own backoff does, and for the same reason: so
# does the cost of having been wrong for that long. Early on, a mistake is minutes of
# cluster time and the gap should be small; four hours in, a bulletin every few minutes
# would be noise charged against a turn each time. The hourly tail is the floor under
# that — an overnight job stays answerable for without ever being chatty, where a
# schedule that simply ran out would leave the longest runs, the ones with most to
# lose, as silent as before.
_CHECKIN_SCHEDULE = (30.0, 120.0, 600.0, 1800.0)
_CHECKIN_INTERVAL = 3600.0


@dataclass
class _Watch:
    """A live background-job watcher: the polling task, and what it is watching.

    The descriptor is kept next to the task because the dispatch guard compares an
    incoming call against the *job's own* ``status_op`` — registry data travelling on
    the descriptor, the same reason the watcher itself can poll generically. Holding
    only the task would have forced the guard to name a tool.

    ``status`` is the last thing the poll learned — ``{state, phase, percent, at}``,
    whatever the status op chose to report, shape-driven like everything else here. A
    check-in reads it instead of polling again: the watcher is already asking, once a
    tick, and a second asker would double every run's status traffic to say what the
    first one already knows. It has to be mutable, which is what a dataclass buys here.
    """
    task: asyncio.Task
    descriptor: dict
    status: dict = field(default_factory=dict)


def _first_line(value: Any) -> str:
    """The first line of an error payload, for a one-line reason. "" when there is none."""
    text = str(value or "").strip()
    return text.splitlines()[0][:200] if text else ""


def _detach_grace() -> float:
    """How long a socket may be gone before this process treats it as gone.

    Not zero, and this is the whole reason the delay exists: reloading a VS Code window
    closes and reopens the socket, and acting on that blink would be a regression — the
    user is right there, and would find their question put away.

    One value, two readers: an unanswered card is set aside past it, and a server no
    conversation has asked to keep stops past it. Both are answering the same question
    — has the window actually gone — so both must answer it the same way.
    """
    try:
        return max(0.0, float(os.environ.get("MIMIR_DETACH_GRACE", "") or 30.0))
    except ValueError:
        return 30.0


class _AgentWorker:
    """Runs MimirAgent in a dedicated background thread with its own event loop.

    Thread-safe queues carry events to the WS layer and approval responses back.
    """

    @classmethod
    def detached(cls, model: str) -> "_AgentWorker":
        """A worker that never builds an agent: no thread, no servers, no LLM wait.

        The stand-in for "this conversation has no agent yet", which is the ordinary state
        until its first query. A real ``_AgentWorker`` rather than a parallel null class, on
        purpose: every getter here answers for ``_agent is None`` (``get_context_mode`` →
        the default its caller passes, ``get_temperature_state`` → the stored preference,
        ``toggles_state`` → empty) and every setter is a no-op in that state. A separate
        class would restate all of those, and then drift from them.

        What it must never be given is a query: ``submit_query`` would put it on a queue no
        loop is reading. The session resolves a real worker through the pool for that.
        """
        worker = object.__new__(cls)
        worker._init_fields(model, None)
        return worker

    def __init__(self, model: str, session_id: str | None = None) -> None:
        self._init_fields(model, session_id)
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

    def _init_fields(self, model: str, session_id: str | None) -> None:
        """Every field, and nothing that starts running.

        Split from ``__init__`` so :meth:`detached` can have the state without the thread,
        the backend wait and the ~19 server spawns.
        """
        self.model = model
        # Whether the overhead `context_overhead_tokens` last returned came from a
        # server's own count. Recorded as it is computed rather than worked out again:
        # answering it needs the system prompt and the tools schema, and the bar asks
        # both questions once a second.
        self._overhead_measured = False
        # The conversation this worker exists for, fixed for its whole life. One worker
        # per session is what lets a turn keep running in a conversation nobody is looking
        # at: a worker shared between conversations can stream only one of them anywhere
        # the user can see.
        #
        # None only for the ends that have no session (tests, standalone construction),
        # where everything session-scoped falls back to the active-session pointer.
        self.session_id: str | None = session_id
        # This conversation's title, for cards raised while the user is reading another.
        # Pushed in by _Session, which is the side that knows what the session is called.
        self.session_title: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._agent: Any = None
        self._ready = threading.Event()
        # Set once this worker's MCP servers are closed, so a caller that asked for the
        # close can wait for the subprocesses to actually be gone.
        self._closed = threading.Event()
        self._error: Exception | None = None

        # Queues for cross-thread communication.
        self.out_q: _queue.Queue[dict] = _queue.Queue()   # agent → WS
        self._approval_q: _queue.Queue[dict] = _queue.Queue()  # WS → approval shim
        self._question_q: _queue.Queue[dict] = _queue.Queue()  # WS → question shim
        self._query_q: _queue.Queue[dict | None] = _queue.Queue()  # WS → query loop
        self._steer_q: _queue.Queue[str] = _queue.Queue()  # WS → running agent (mid-run steering)
        # Length of the history the running turn was handed (see _run_query).
        self._turn_submitted_len: int | None = None
        # When this process last had a client listening, or None while one does. Written
        # by the pump on the event loop and read by this worker's thread, which is why
        # it is a plain float and not an asyncio primitive: the thread must never touch
        # loop state. See _unattended_past_grace.
        self.unattended_since: float | None = None
        # True while a turn of this conversation is set aside awaiting an answer. Read
        # by the pool, which must not reap the agent out from under it.
        self.has_deferral: bool = False
        # The card a parked turn is waiting on, set for exactly as long as it waits.
        # Read by a connection that arrives while the wait is on (see _emit_prompt).
        self._pending_prompt: dict | None = None
        # Event signalled whenever a new item is placed on _query_q so the
        # background loop wakes up immediately instead of waiting out the poll interval.
        self._query_event = threading.Event()

        self._current_task: asyncio.Task | None = None
        # The session the front-end is currently showing. Not the same question as
        # ``session_id``: this worker always works for its own session, and this says
        # whether that session is the one on screen. Kept in sync by _Session.
        self.active_session_id: str | None = None
        # Session a running query belongs to, captured when it starts. Events carry the
        # session they were produced for, so a turn running in a conversation that is not
        # on screen can be told apart from the one that is — and routed to its own
        # transcript rather than streamed into whatever the user is reading.
        self._query_session_id: str | None = None
        # Background-job watchers: job_key -> _Watch(task, descriptor), polling a
        # detached run to completion. Registered by the agent loop via _register_bg_job
        # (below) and read back by _watched_bg_jobs, which the dispatch guard uses.
        self._bg_jobs: dict[str, _Watch] = {}
        # The check-in cycle covering every job of this conversation, or None between
        # waves. One per worker rather than one per job: a worker *is* a conversation,
        # so three jobs launched in the same step share a schedule and report together
        # instead of waking it three times a minute apart.
        self._checkin_task: asyncio.Task | None = None
        # Set by the front-end when the user leaves a conversation whose turn is parked
        # on them: every wait of that turn returns at once instead (query_engine.deferral).
        self._defer = threading.Event()
        # The answer a resume turn carries, handed to the first prompt of its kind
        # instead of putting the card up again: ``{"type", "response"}``.
        self._preanswer: dict | None = None
        # The raw questions of the pending question card — how a deferred question is
        # matched to the call that asked it.
        self._pending_questions: list | None = None
        # Cards whose wait gave up before an answer came. An answer that lands after
        # that is dropped rather than queued: the call that asked has already moved on,
        # and a queued answer would settle whichever prompt comes next instead.
        self._expired_prompt_ids: set[str] = set()

        self._thread: threading.Thread | None = None

    # ── Background thread ─────────────────────────────────────────────────────

    def _main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._live())
        finally:
            loop.close()

    async def _live(self) -> None:
        """Connect, serve queries, close the servers — all in one task.

        One task, not three, because of where ``stdio_client`` keeps its cancel scope:
        anyio anchors it to the task that entered the context, and exiting it anywhere
        else raises ``RuntimeError: Attempted to exit cancel scope in a different task``.
        The exit stack is then left half-unwound — the streams closed, the subprocess
        alive — so the next tool call of a turn still running on this agent fails with
        ``ClosedResourceError`` and no server is reaped. Setup, the query loop and the
        close therefore share the task that owns the stack, and the close runs when the
        query loop has returned, which is when no turn can be in flight.
        """
        await self._setup()
        try:
            if self._error is None:
                await self._query_loop()
        finally:
            await self._close_agent()

    async def _close_agent(self) -> None:
        """Close the agent's MCP servers. Runs in :meth:`_live`, which opened them."""
        agent, self._agent = self._agent, None
        if agent is None:
            self._closed.set()
            return
        try:
            await agent.cleanup()
        except Exception:
            logger.warning("worker %s: closing MCP servers failed",
                           self.session_id or "<no session>", exc_info=True)
        finally:
            self._closed.set()

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
        # Holds the agent for as long as it is half-built, so a failure partway through
        # the ~19 connects still has something to close. ``self._agent`` cannot serve
        # that: every getter reads it, and it means "ready", not "under construction".
        building: Any = None
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

            agent = building = MimirAgent(model=self.model, session_id=self.session_id)
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
            if self._agent is None and building is not None:
                # Whatever connected before the failure is holding subprocesses, and
                # this task is the only one that may close them.
                try:
                    await building.cleanup()
                except Exception:
                    logger.warning("worker %s: closing a half-built agent failed",
                                   self.session_id or "<no session>", exc_info=True)
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
            self._query_session_id = item.get("session_id") or self._own_session()
            # What this turn was handed, so whoever writes its answer can tell the
            # turn's own messages from the prefix it inherited. Recorded here because
            # this is where the turn begins, and carried on the answer rather than kept
            # per socket: a turn can outlive the socket that submitted it, and a
            # boundary held there is one nobody has when the answer lands.
            self._turn_submitted_len = len(item.get("history") or [])
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
        tid = threading.get_ident()
        _ROUTER.register(tid, lambda text: self.out_q.put({"type": "output", "text": text}))

        cancelled = False
        try:
            # The prose of the streamed block, kept so it can be *recorded*. The
            # deltas themselves are not: hundreds per turn, and a journal of them is
            # unreadable. But without an aggregate the record keeps only the tool rows
            # and the final answer, so a turn read back after an absence has lost
            # everything the agent said between its tools — which is most of what makes
            # a turn legible. Flushed at each boundary, in order, so the replay
            # interleaves prose and tools the way the live stream did.
            text_buf: list[str] = []

            def _flush_text() -> None:
                if not text_buf:
                    return
                text = "".join(text_buf)
                text_buf.clear()
                self.emit_assistant_text(text)

            def _token_cb(delta: str) -> None:
                text_buf.append(delta)
                self.out_q.put({"type": "token", "text": delta})

            # Reasoning text of the streaming block, kept so its size can be reported
            # in tokens when the block closes. Counted here, not in the webview, so the
            # number comes from the same tokenizer as the context bar.
            think_buf: list[str] = []

            def _think_token_cb(delta: str) -> None:
                think_buf.append(delta)
                self.out_q.put({"type": "thinking", "text": delta})

            def _think_start_cb() -> None:
                _flush_text()      # prose written before this block belongs above it
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
                # Before the structured event, which is what puts the prose above the
                # tools it preceded rather than after them.
                _flush_text()
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

        # The answer carries the closing block in full, so whatever is still buffered
        # is a copy of it: dropped rather than flushed, or the replay shows it twice.
        try:
            text_buf.clear()
        except NameError:
            pass
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
            # This turn's agent, not whichever one is on screen: several conversations
            # answer at once, and their modes need not agree.
            answer_ev["_context_mode"] = getattr(self._agent, "context_mode", "full")
        answer_ev["_submitted_len"] = getattr(self, "_turn_submitted_len", None)
        deferred = self._deferred_record()
        if deferred is not None and not cancelled:
            answer_ev["_deferred"] = deferred
        # Steering the loop never read. It is drained at a step boundary, and the step
        # that produces the final answer has none after it: a message typed while that
        # answer streams lands in the queue after the loop has stopped looking. Taken
        # off the queue here, so it cannot bleed into whatever runs next, and handed to
        # the session, which starts a turn for it — unanswered, it is a message the user
        # sees waiting for a run that has ended. Dropped after a cancel: the turn it was
        # aimed at is abandoned, and so is the instruction aimed at it.
        unread = self._drain_steer_q()
        if unread and not cancelled:
            answer_ev["_unconsumed_steer"] = unread

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
            # What the user is about to be shown, recorded so a later revert can tell
            # whether the file still holds it. Anything else on disk by then was written
            # by something outside this review — another session, or the user's editor.
            agent.approvals.note_reviewed(path, current)
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

    def _own_session(self) -> str | None:
        """The conversation this worker works for.

        ``session_id`` when it has one, else the session on screen. The fallback is for
        a worker built before any session was known — which is how the single shared
        worker was built, and is still how a bare one in a test is. Reading the on-screen
        session is only ever right for such a worker: one with a session of its own must
        use it, or it writes another conversation's checklist the moment the user looks
        somewhere else.
        """
        return self.session_id or self.active_session_id

    def _load_todos(self) -> list:
        try:
            try:
                from ...prompt.system_prompt import _load_todo_items
            except ImportError:
                from mimir.client.prompt.system_prompt import _load_todo_items
            return _load_todo_items(_todo_file_for_session(self._own_session()))
        except Exception:
            return []

    def _clear_todos(self) -> None:
        """Wipe this worker's own session's todo file."""
        try:
            todo_file = _todo_file_for_session(self._own_session())
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
        self.has_deferral = False     # answered: the turn is nobody's debt any more
        self._query_event.set()

    def _emit_prompt(self, payload: dict, questions: list | None = None) -> dict:
        """Send a card the turn is about to park on, and remember it while it waits.

        An agent outlives the connections that read it: a socket that drops while it is
        parked leaves the card on a client nobody is holding, and the turn waiting on an
        answer nobody can give. This agent's query loop is serial, so every later query of
        *this conversation* queues behind that wait and it reads as hung, with nothing on
        screen to explain it. Kept here, the card goes back in front of whoever reconnects
        (``_Session._resend_parked_prompt``).
        """
        # Which conversation is asking. Structured, never spelled into the label or the
        # question text: that text is also what the model sees, so a marker written there
        # lands in the conversation's own history. The client renders the attribution from
        # these and — the part that is not cosmetic — sends them back on the answer, which
        # is how the answer reaches the agent that asked rather than whichever one is on
        # screen.
        payload = {
            **payload,
            "session_id": self._query_session_id or self.session_id,
            "session_title": self.session_title or "",
        }
        self._pending_prompt = dict(payload)
        self._pending_questions = questions
        # Nobody is there to read it (deferring), or the answer is already in hand
        # (resuming): the wait below settles it without a card.
        if self._deferring() or self._preanswer_for(payload) is not None:
            return payload
        self.out_q.put(payload)
        # The attributed payload, so a caller that has to take the card back down
        # (an expired question) addresses the same conversation the card named.
        return payload

    def emit_assistant_text(self, text: str) -> bool:
        """Record one streamed prose block. True when it was worth recording.

        The one rule in the aggregation, and the reason it is a method rather than a
        line inside the streaming closure: whitespace between a tool result and the
        next call is not something the agent *said*, and a journal of empty blocks
        would put a blank bubble in every replayed turn.
        """
        if not text or not text.strip():
            return False
        self.out_q.put({"type": "assistant_text", "text": text})
        return True

    def _unattended_past_grace(self) -> bool:
        """Whether this process has had no client for longer than the grace period.

        The criterion is *attached*, not *elapsed*: a card with somebody there waits
        for ever, which is deliberate and what ``test_approval_wait`` pins. What
        changes when nobody is there is not how long the wait may be but whether there
        is anything to wait for.
        """
        # Absent until the pump's first tick reaches this worker, which is a real
        # state and not only a test's: the safe reading of no information is that
        # somebody is there, because the cost of being wrong the other way is a card
        # put away under the user's nose.
        since = getattr(self, "unattended_since", None)
        if since is None:
            return False
        return (time.monotonic() - since) >= _detach_grace()

    def set_non_interactive(self, value: bool = True) -> None:
        """Tell the agent there is no terminal behind it.

        Read by the policy engine's ``_is_interactive_session``, which otherwise has to
        infer it from the tty — and a detached server's stdout is a log file, which is
        not a tty but also not proof that nobody is reachable. Saying so plainly is
        what keeps the interactive path from being attempted at all.
        """
        agent = self._agent
        if agent is not None:
            agent.non_interactive = bool(value)

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

    def _await_response(
        self, q: "_queue.Queue[dict]", timeout: float | None = None
    ) -> dict | None:
        """Block until a WS response lands on ``q``, or until *timeout* passes.

        An unanswered approval must keep the agent *parked*: it must never silently
        proceed just because the user was slow to respond, so approvals pass no
        timeout and wait indefinitely. To stay responsive to the Stop button, we poll
        in short slices and bail the moment the agent's cancel flag is set (from the
        WS thread), returning ``None`` for cancelled.

        With a *timeout*, the wait gives up once it passes and returns
        :data:`TIMED_OUT` — told apart from the cancelled ``None`` because the caller
        has a card on screen to take back down, and the two endings read differently
        to the model.

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
        deadline = None if timeout is None else time.monotonic() + timeout
        try:
            with human_pause.human_pause():
                while True:
                    agent = self._agent
                    if agent is not None and agent._cancel_flag.is_set():
                        return None
                    if self._deferring():
                        self._record_deferral()
                        return None
                    if deadline is not None and time.monotonic() >= deadline:
                        return TIMED_OUT
                    try:
                        return q.get(timeout=0.25)
                    except _queue.Empty:
                        pass
                    # Only once the queue has had its turn and come back empty, and
                    # this order is the point: an answer already in hand must win over
                    # setting the card aside, or a reply that crossed with the grace
                    # elapsing is thrown away and the user answered into a void.
                    #
                    # Nobody to answer, for long enough that it is not a window
                    # reload. Set aside rather than waited out: the wait has no
                    # timeout by design, so without this a detached run meets its
                    # first sensitive tool and holds this thread for ever — and the
                    # pool then reaps the agent out from under it.
                    #
                    # Only the indefinite waits. A question carries its own five
                    # minutes and a documented meaning for running out of them
                    # ("go ahead with what you recommend"); overriding that here
                    # would change an answer the model has already been promised.
                    if deadline is None and self._unattended_past_grace():
                        self._defer.set()
        finally:
            # Answered, cancelled or raised through: the turn is not parked any more,
            # and a card resent past this point would be one nothing is waiting on.
            self._pending_prompt = None
            self._pending_questions = None

    def _expire_prompt(self, req_id: str, q: "_queue.Queue[dict]") -> None:
        """Note that *req_id*'s wait gave up, and clear any answer already in flight.

        The queue is drained for an answer that crossed the wall — the user clicking as
        the wait ended — because nothing reads it any more and the next prompt would.
        Later arrivals are refused by id in ``resolve_question``.
        """
        self._expired_prompt_ids.add(req_id)
        # Bounded: the ids are uuids and never recur, so none ever leaves on its own.
        if len(self._expired_prompt_ids) > 200:
            self._expired_prompt_ids = {req_id}
        while True:
            try:
                q.get_nowait()
            except _queue.Empty:
                break

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
        # The pool reads this: a deferred turn has cleared `_pending_prompt`, so
        # `is_parked()` is False and nothing else would stop the agent being released
        # while the user still owes it an answer.
        self.has_deferral = True

    def _approval_shim(
        self, tool_name: str, arguments: dict, max_attempts: int = 3
    ) -> tuple[bool, str]:
        """Sync approval — blocks background thread until WS client responds.

        The WS event loop (main thread) is unaffected; it forwards the approval
        prompt to the client and puts the response in _approval_q.
        """
        from ...context.capabilities import (
            kind_for, label_for, preview_spec, reversibility_of,
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
            # The tool's work family, so the card is drawn with the same icon as the
            # activity row for the same call. Read off the registry like everything else
            # here — the front-end must not learn which tool happens to be a shell.
            "kind": kind_for(tool_name, agent.tool_caps),
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
        from ...context.capabilities import IRREVERSIBLE, kind_for, label_for
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
            # Same family (and so the same icon) as the row for this call.
            "kind": kind_for(tool_name, agent.tool_caps),
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

    def _question_shim(self, questions: list, timeout_secs: float | None = None) -> dict:
        """Sync clarification questions — blocks the worker thread until answered.

        Mirrors ``_approval_shim``: emits a ``user_question`` card carrying the whole
        batch of questions to the client and blocks the agent worker thread (not the
        WS event loop) until a ``user_question_response`` arrives. The frontend shows
        the questions one at a time and returns all ``answers`` together. A cancel
        returns no answers so the agent proceeds with its best judgment.

        ``timeout_secs`` bounds the wait (``ask_user_question`` passes it; plan
        approval does not). When it passes, a ``prompt_expired`` event takes the card
        off the client's screen — it is answering a question nobody is waiting on any
        more — and the result says ``timed_out``, which the tool turns into "nobody
        answered, go with what you recommended".
        """
        req_id = str(uuid.uuid4())
        emitted = self._emit_prompt({
            "type": "user_question",
            "id": req_id,
            "questions": list(questions),
            # So the card can show what is left of the wait. Closing it is this side's
            # call (``prompt_expired``); the client only displays the countdown, which
            # is what keeps the card disappearing from reading as a glitch.
            "timeout_secs": int(timeout_secs) if timeout_secs else None,
        }, questions=list(questions))
        response = self._await_response(self._question_q, timeout=timeout_secs)
        if response is TIMED_OUT:
            self._expire_prompt(req_id, self._question_q)
            self.out_q.put({
                "type": "prompt_expired",
                "id": req_id,
                "kind": "user_question",
                "session_id": emitted.get("session_id"),
                "session_title": emitted.get("session_title") or "",
                "timeout_secs": int(timeout_secs or 0),
            })
            return {"answers": [], "timed_out": True, "timeout_secs": timeout_secs}
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

    def has_work_pending(self) -> bool:
        """True while a turn is running here *or* queued to run. What eviction reads.

        ``is_busy`` answers a different question — is there a run to steer into — and is
        False for the stretch between a turn being submitted and the query loop picking it
        up. Releasing the agent in that gap closes the MCP servers under a turn that is
        about to start, and every tool call it makes then fails on a dead stream.
        """
        return self._current_task is not None or not self._query_q.empty()


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


    def _drain_steer_q(self) -> list[str]:
        """Pop and return all queued steer messages (the agent's ``_poll_steer``)."""
        out: list[str] = []
        while True:
            try:
                out.append(self._steer_q.get_nowait())
            except _queue.Empty:
                break
        return out

    def _register_bg_job(self, descriptor: dict, owner: str | None = None) -> bool:
        """Register a completion watcher for a detached run (the agent's hook).

        Called on the worker loop from the agent's tool dispatch. Dedups on
        ``job_key`` so re-launching the same job never spawns a second watcher.
        Returns True when a watcher is (already) active — the loop uses this to
        tell the model it may end its turn.

        The session that launched the job is captured here and travels with the
        watcher: a two-hour build outlives the conversation on screen, and the wake
        belongs to the conversation that asked for it, not to whichever one the user
        happens to be reading when it lands.

        *owner* names that session explicitly. The running query is the right answer
        when the agent registers a job it has just launched, and the wrong one when a
        watcher is being put back on a run found on disk — there is no running query
        then, and the session is whichever directory the descriptor came out of.
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
        session_id = owner or self._query_session_id or self._own_session()
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
        task.add_done_callback(self._watcher_died)
        self._bg_jobs[job_key] = _Watch(task, descriptor)
        logger.info("background job %r of session %s is now watched by %r",
                    job_key, session_id, (descriptor.get("status_op") or {}).get("tool"))
        self._start_checkins(session_id)
        return True

    def rearm_detached_jobs(self) -> dict:
        """Put watchers back on this session's runs, and report the ones that ended.

        What a restart takes away is the *promise to report* a run, never the run: it
        keeps going in its own session directory, indifferent. So this reads the
        descriptors off disk and makes the promise again — a watcher for everything
        still going, and a ``job_complete`` for everything that finished while nothing
        was listening, which is the wake its conversation has been owed since.

        Only runs nothing is already watching. ``_bg_jobs`` is the authority on that,
        and the ordinary case — a server that simply stayed up — finds it full and does
        nothing at all. Returns what it did, for the log and for tests.
        """
        session_id = self._own_session()
        if not session_id:
            return {"rearmed": [], "reported": []}
        jobs = scan_session(session_id)
        if not has_baseline(session_id):
            # First time this session has been looked at: everything already finished
            # was reported by whatever was watching it, long before markers existed.
            # Claiming that history as a backlog of wakes would wake a conversation for
            # every build it ever ran.
            establish_baseline(session_id, jobs)
            jobs = [job for job in jobs if job.live]
        rearmed: list[str] = []
        reported: list[str] = []
        for job in jobs:
            existing = self._bg_jobs.get(job.job_key)
            if existing is not None and not existing.task.done():
                continue   # a watcher is already holding it
            if job.live:
                if self._register_bg_job(job.descriptor(), owner=session_id):
                    rearmed.append(job.job_key)
                continue
            # Ended with nobody watching. Emitted as the watcher would have — the
            # descriptor's own server and kind included — because a consumer must not be
            # able to tell a run reported late from one reported on time, and the wake
            # text names the kind. Settled by that consumer, not here: see
            # ``job_scan.mark_wakes_reported``.
            descriptor = job.descriptor()
            self.out_q.put({
                "type": "job_complete",
                "job_key": job.job_key,
                "server": descriptor.get("server"),
                "kind": descriptor.get("kind"),
                "state": job.state,
                "session_id": session_id,
                "summary": {"command": job.command, "exit_code": job.exit_code},
                "status_op": job.status_op(),
            })
            reported.append(job.job_key)
        if rearmed or reported:
            logger.info("job scan: session %s re-armed %d run(s) and reported %d "
                        "that had ended", session_id, len(rearmed), len(reported))
        return {"rearmed": rearmed, "reported": reported}

    @staticmethod
    def _watcher_died(task: asyncio.Task) -> None:
        """Report a watcher or check-in cycle that ended on an exception.

        Nothing awaits these tasks, so an exception escaping one is held until the GC
        notices and prints "Task exception was never retrieved" — long after the run it
        was holding finished into silence. The failure and the missing wake are the same
        event, and this is the only place they can be named together.
        """
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.warning("background-job task died: %r — any run it was holding will "
                           "not wake anyone", exc, exc_info=exc)

    def _start_checkins(self, session_id: str | None) -> None:
        """Begin a check-in cycle for this conversation, unless one is already running.

        Called on every registration, and a no-op for all but the first of a wave: a job
        joining a cycle in flight inherits what is left of it rather than restarting the
        clock, which is what keeps three jobs launched together to one bulletin apiece
        instead of three.

        What decides is whether this registration *starts* a wave — whether it is the
        only live run of the conversation. The cycle cannot end on its own the instant
        the last job does: its tail is hourly, so it is asleep for up to an hour after
        a wave finishes, and a job launched into that gap would inherit the remainder
        of a schedule that no longer describes anything and wait out the hour for its
        first bulletin. A new wave therefore replaces the cycle rather than joining it.
        """
        wave_starts = len(self.watched_job_keys()) <= 1
        if self._checkin_task is not None and not self._checkin_task.done():
            if not wave_starts:
                return
            self._checkin_task.cancel()
        try:
            self._checkin_task = asyncio.get_running_loop().create_task(
                self._checkin_cycle(session_id))
        except RuntimeError:
            # Same precondition as the watcher's, and the same honesty: without a loop
            # there are no check-ins, and the run is still watched. Not a failure of the
            # registration — the completion wake does not depend on this.
            logger.warning("background check-ins unavailable for session %s: no running "
                           "event loop to host the cycle", session_id)
            self._checkin_task = None
            return
        self._checkin_task.add_done_callback(self._watcher_died)

    async def _checkin_cycle(self, session_id: str | None) -> None:
        """Emit ``job_checkin`` down ``_CHECKIN_SCHEDULE``, then hourly while runs last.

        Reports what the watchers have already seen — ``_Watch.status``, written by the
        poll that is running anyway — so a check-in costs no status traffic at all. The
        turn it may start is the only cost, and the session decides whether to spend it.

        Ends the moment this conversation has no live run left: past that the completion
        wakes have landed and there is nothing to be reassured about. That is the only
        thing that ends it — the tail does not run out — so an overnight run is still
        answered for at hour six, where a schedule with a last point would have gone
        quiet exactly where the stakes were highest.
        """
        try:
            for delay in chain(_CHECKIN_SCHEDULE, repeat(_CHECKIN_INTERVAL)):
                await asyncio.sleep(delay)
                jobs = [
                    {"job_key": key, "kind": w.descriptor.get("kind"),
                     "server": w.descriptor.get("server"), **w.status}
                    for key, w in self._bg_jobs.items() if not w.task.done()
                ]
                if not jobs:
                    return
                self.out_q.put({
                    "type":       "job_checkin",
                    "session_id": session_id,
                    "jobs":       jobs,
                })
        finally:
            # Only if the slot still holds *this* cycle. A new wave cancels the old one
            # and puts its own task in the slot, and cancellation is delivered after
            # that — so an unconditional clear here would erase the live cycle's
            # reference and let the next launch start a second one beside it.
            if self._checkin_task is asyncio.current_task():
                self._checkin_task = None

    def watched_job_keys(self) -> list[str]:
        """Keys of the runs this conversation still has in flight.

        Asked before delivering a check-in it held back — a bulletin on runs that have
        since finished is worse than none — and by the panel, which reports a live run
        beside a live turn so walking away from one is a question rather than a
        surprise.

        A worker with no table of watchers has no watchers, which is the honest answer
        to give rather than an attribute error: this is read for every row of every
        session listing, behind a guard that would blank the whole panel.
        """
        watched = getattr(self, "_bg_jobs", None) or {}
        return [k for k, w in watched.items() if not w.task.done()]

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
                    raw = await asyncio.wait_for(
                        self._agent._run_tool(
                            status_tool, dict(status_op.get("args") or {}),
                            record_observations=False),
                        timeout=_TOOL_CALL_TIMEOUT)
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
                except asyncio.TimeoutError:
                    # Named rather than lumped in with the failures below: "it never
                    # answered" and "it answered with an error" are the two things a
                    # reader most needs to tell apart, and only this knows which it saw.
                    payload = {}
                    state, reason = "", (f"'{status_tool}' did not answer within "
                                         f"{_TOOL_CALL_TIMEOUT:.0f}s")
                    logger.warning("background job %r: its status op %r did not answer "
                                   "within %.0fs", job_key, status_tool,
                                   _TOOL_CALL_TIMEOUT)
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
                # What a check-in reports, written by the poll that is running anyway.
                # The state goes with it: "running" and "its status stopped being
                # readable" are the two things a bulletin most needs to tell apart, and
                # only this loop knows which one it is looking at.
                watch = self._bg_jobs.get(job_key)
                if watch is not None:
                    watch.status = {"state": state or "unreadable", "phase": phase,
                                    "percent": percent, "at": time.monotonic()}
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
                # Under the same deadline as the status poll, and here it matters more:
                # the run is already over, so a summary op that never answers holds back
                # the wake itself. Better a wake with nothing in it than no wake.
                raw = await asyncio.wait_for(
                    self._agent._run_tool(
                        summary_tool, dict(summary_op.get("args") or {}),
                        record_observations=False),
                    timeout=_TOOL_CALL_TIMEOUT)
                # parse_tool_payload for the same reason the status tick uses it: a
                # tool result is an envelope, its text blocks and any annotation
                # appended after them, and a bare load reads that as one broken
                # document. A summary lost here reaches the model as "it recorded no
                # result of its own" — the wake still lands, emptied of the thing it
                # was carrying.
                summary = parse_tool_payload(raw) if isinstance(raw, str) else (raw or {})
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                summary = {}
                logger.warning("background job %r finished, but its summary op %r did "
                               "not answer within %.0fs; waking it without one",
                               job_key, summary_tool, _TOOL_CALL_TIMEOUT)
            except Exception:
                summary = {}

        logger.info("background job %r of session %s finished (%s); waking it",
                    job_key, session_id, state)
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
        # Deliberately not marked as reported here. Emitting is not delivering: the
        # event still has to be folded into a turn by whatever consumes the bus, and a
        # run settled at the moment its event is queued is one every later scan takes
        # for already delivered — so a wake nobody has read is a wake nobody ever will.
        # The consumer writes the marker — see ``job_scan.mark_wakes_reported``.
        self._bg_jobs.pop(job_key, None)

    def resolve_approval(self, choice: str, approved_files: list | None = None) -> None:
        self._approval_q.put({"choice": choice, "approved_files": approved_files})

    def resolve_question(self, answers: list | None, prompt_id: str | None = None) -> None:
        if prompt_id and prompt_id in self._expired_prompt_ids:
            logger.info("dropping an answer to question %s, whose wait had expired",
                        prompt_id)
            return
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

    def get_context_mode(self, default: str = "full") -> str:
        """The agent's context mode, or *default* while there is no agent.

        The default is the caller's to give, and it matters: the context budget is
        derived from this, and answering "compact" for a conversation that was running
        in full mode sized its window at 32k. A session resumed before its first query
        passes the mode it was saved with; everything else gets ``MimirAgent``'s own
        default, which is what the agent built for it will have.
        """
        if self._agent is not None:
            return getattr(self._agent, "context_mode", default)
        return default

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
        every prompt — should not have to ask the model to do it. The alternative is deleting
        files under a store whose path they have no reason to know.

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

        Reads what :meth:`context_overhead_tokens` recorded on its last call, so ask it
        after that one. A figure read back from the calibration cache counts as measured:
        a server did measure it, against this very system prompt and tool set.
        """
        return self._overhead_measured

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

        A measured figure outlives the process too: it is written to the calibration
        cache under a fingerprint of the whole fixed part (model, mode, system prompt,
        advertised tools) and read back on the next run. A session reopened against a
        fresh server is therefore accounted for exactly from its first frame rather
        than from its first answer — which is what kept the bar reading over-full on a
        resume. The fingerprint is the safety: change a mode, a server or the prompt
        and the entry is simply not found, so the estimate answers instead of a stale
        measurement wearing its authority.

        Best-effort, cached via the backend's token cache; never raises.
        """
        agent = self._agent
        if agent is None:
            return 0
        try:
            from ...query_engine.backends.factory import get_backend
            from ...query_engine import token_calibration
            from ...agent_core import build_base_system_content
            backend = get_backend()
            # getattr, not a direct call: a backend stub without the calibration
            # should fall through to the estimate, not lose the overhead entirely
            # to the except below.
            probe = getattr(backend, "measured_prompt_overhead", None)
            measured = probe(self.model) if callable(probe) else None
            mode = getattr(agent, "mode", "") or ""
            build = getattr(agent, "build_system_content_now", None)
            prompt = build(mode) if callable(build) else build_base_system_content()
            narrow = getattr(agent, "advertised_tools_for_mode", None)
            tools = narrow(mode) if callable(narrow) else getattr(agent, "tools", None)
            tools_json = json.dumps(tools) if tools else ""
            key = token_calibration.overhead_key(
                self.model, getattr(agent, "context_mode", "full"), prompt, tools_json
            )
            if measured is not None:
                # Write-through, so the next run of this same configuration starts
                # calibrated. A no-op when the figure is already on disk, which it is
                # for all but the first call after a turn — the bar asks every second.
                token_calibration.remember_overhead(key, measured)
                self._overhead_measured = True
                return measured
            remembered = token_calibration.recall_overhead(key)
            if remembered is not None:
                # Measured by a server too, against this very prompt and tool set.
                self._overhead_measured = True
                return remembered
            self._overhead_measured = False
            total = backend.count_text_tokens(self.model, prompt, allow_network=False)
            if tools_json:
                total += backend.count_text_tokens(
                    self.model, tools_json, allow_network=False
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
        here rather than at the dozens of emit sites.

        An event produced outside a query — the ``ready`` the setup emits, an error from
        it — carries this worker's own session rather than ``None``. It belongs to that
        conversation as much as a turn's own output does, and a worker building lazily
        emits it while the user may well be reading a different one.
        """
        events: list[dict] = []
        while True:
            try:
                ev = self.out_q.get_nowait()
            except _queue.Empty:
                break
            if isinstance(ev, dict):
                ev.setdefault("session_id", self._query_session_id or self.session_id)
            events.append(ev)
        return events

    def shutdown(self) -> None:
        # Cancel any in-flight background-job watchers, and the check-in cycle over
        # them, on the worker loop.
        tasks = [w.task for w in self._bg_jobs.values()]
        if self._checkin_task is not None:
            tasks.append(self._checkin_task)
        for task in tasks:
            if self._loop is not None and not task.done():
                self._loop.call_soon_threadsafe(task.cancel)
        self._query_q.put(None)
        self._query_event.set()

    def aclose(self, timeout: float = 20.0) -> None:
        """Shut the worker down AND close the agent's servers, then join the thread.

        ``shutdown`` only ends the query loop. ``agent.exit_stack`` is where the ~19 MCP
        server subprocesses are held, and a worker released when its conversation goes
        quiet must close it, or each release strands a full set of servers and a few hours
        of use exhausts the machine rather than freeing anything.

        The close itself belongs to :meth:`_live`, the task that opened those servers;
        this asks for it and waits. ``stdio_client`` is an anyio context whose cancel
        scope is anchored to the entering task, and exiting it from any other task raises
        and leaves the stack half-unwound: streams closed, subprocess alive, and a turn
        still running on that agent failing its next tool call with
        ``ClosedResourceError``. So the sentinel is the whole mechanism — the query loop
        returns, and its own ``finally`` closes the servers.

        A turn in flight is cancelled first. Without that the sentinel waits behind it:
        shutdown would cost the rest of a turn per conversation, and a turn allowed to
        run into the close would be the one that hits the dead streams.

        Best-effort and bounded: a server wedged in its own shutdown must not hold the
        pool, and the subprocess dies with this process in the worst case.
        """
        loop = self._loop
        if self._agent is None or loop is None or loop.is_closed():
            self.shutdown()
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            return
        self.cancel()        # no-op when nothing is running
        self.shutdown()      # the sentinel: ends the query loop, whose finally closes
        if not self._closed.wait(timeout):
            # The caller is already free; this worker's thread is left to finish on its
            # own, and the OS reaps what is left when the process exits.
            logger.warning("worker %s: MCP servers did not close within %.1fs; letting "
                           "its thread finish on its own",
                           self.session_id or "<no session>", timeout)
            return
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)

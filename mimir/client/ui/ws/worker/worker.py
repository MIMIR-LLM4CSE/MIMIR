"""The background agent worker for the WebSocket server.

``AgentWorker`` runs MimirAgent in a dedicated thread with its own asyncio event
loop so blocking approval prompts never freeze the WebSocket event loop. It talks
to the connections exclusively through thread-safe queues. The wire protocol is
documented in ``server.main``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import difflib
import json
import logging
import os
import queue as _queue
import threading
from typing import Any, Callable

from .....servers._shared.embed import verify_ssl
from .....servers._shared.responses import err
from ....agent_core import MimirAgent, build_base_system_content
from ....config import TEMPERATURE_BACKENDS
from ....config.models import thinking_can_disable, thinking_profile
from ....config.preferences import load_temperature
from ....context.capabilities import panel_sections as _declared
from ....context.todo import load_todo_items
from ....core.settings import read_setting
from ....extensions import all_servers
from ....protocol.tool_result import normalize_tool_content
from ....query_engine import token_calibration
from ....query_engine.backends.factory import get_backend
from ....tool_execution.direct import NO_AGENT, call_tool_directly

# Import the shared runtime FIRST so its cwd bootstrap runs before config.paths
# (and the backend factory) capture the workspace root at import time.
from ..runtime import (
    _ORIGINAL_STDOUT,
    _ROUTER,
    _todo_file_for_session,
    augment_query_with_resources,
)

# After the runtime: it reads the workspace root that the bootstrap above pins.
from .human import HumanBridge
from .jobs import JobWatcher

logger = logging.getLogger(__name__)

# The backend is the server's, not a conversation's: once one agent has seen it answer,
# the next conversation's agent does not wait on it again.
_backend_seen_up = False


class _Outbox(_queue.Queue):
    """A worker's events on their way to the pump, which each one wakes.

    The pump used to look every 50 ms whether or not anything was there. ``on_put`` is
    set by the pool once the worker is built, to a thread-safe nudge of the pump; until
    then, and if the loop is gone, the pump's own tick still drains what is here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.on_put: Callable[[], None] | None = None

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        notify = self.on_put
        if notify is not None:
            try:
                notify()
            except RuntimeError:
                pass   # the loop is closing; there is no pump left to wake


class AgentWorker:
    """Runs MimirAgent in a dedicated background thread with its own event loop.

    Thread-safe queues carry events to the WS layer and approval responses back.
    """

    @classmethod
    def detached(cls, model: str) -> "AgentWorker":
        """A worker that never builds an agent: no thread, no servers, no LLM wait.

        The stand-in for "this conversation has no agent yet", which is the ordinary state
        until its first query. A real ``AgentWorker`` rather than a parallel null class, on
        purpose: every getter here answers for ``_agent is None`` (``setting("context_mode")`` →
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
        # Pushed in by Connection, which is the side that knows what the session is called.
        self.session_title: str = ""
        self._loop: asyncio.AbstractEventLoop | None = None
        self._agent: Any = None
        self._ready = threading.Event()
        # Set once this worker's MCP servers are closed, so a caller that asked for the
        # close can wait for the subprocesses to actually be gone.
        self._closed = threading.Event()
        self._error: Exception | None = None

        # Queues for cross-thread communication.
        self.out_q = _Outbox()                               # agent → WS
        self._query_q: _queue.Queue[dict | None] = _queue.Queue()  # WS → query loop
        self._steer_q: _queue.Queue[str] = _queue.Queue()  # WS → running agent (mid-run steering)
        # The person behind this agent: the cards it raises and their answers.
        self.human = HumanBridge(self)
        # Length of the history the running turn was handed (see _run_query).
        self._turn_submitted_len: int | None = None
        # Set (thread-safely, on this worker's loop) whenever an item is placed on
        # _query_q: the query loop waits on it rather than polling the queue.
        self._query_wake: asyncio.Event | None = None

        self._current_task: asyncio.Task | None = None
        # Session a running query belongs to, captured when it starts. Events carry the
        # session they were produced for, so a turn running in a conversation that is not
        # on screen can be told apart from the one that is — and routed to its own
        # transcript rather than streamed into whatever the user is reading.
        self._query_session_id: str | None = None
        # The runs this conversation left going, and the check-ins about them.
        self.jobs = JobWatcher(self)
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

        The request is the one the agent actually depends on — the backend's
        ``readiness_urls`` (the model list for every OpenAI-compatible endpoint,
        ``/api/tags`` for Ollama, nothing for a hosted API) — asked with the same
        client, proxy posture and TLS policy the backends themselves use. A probe that asks a different question of a different client
        than the chat path can fail where the chat path succeeds, which is exactly
        what a readiness check must never do.

        A second URL is a fallback (a local vLLM's ``/health`` answers while the engine
        is still loading weights, before ``/v1/models`` does). It is only a fallback
        because it lives at the server root, which an ingress route in front of the
        endpoint usually does not expose.

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



        # Which endpoint, and what to ask it, is the backend's to say: a hosted API has
        # nothing to wait on, a self-hosted server names what to poll.
        backend = os.environ.get("LLM_BACKEND", "vllm")
        urls = get_backend().readiness_urls()
        if not urls:
            return
        health_url, fallback_url = urls[0], (urls[1] if len(urls) > 1 else "")

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
                logger.debug("_get: a best-effort step failed and was skipped", exc_info=True)
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
        global _backend_seen_up
        building: Any = None
        try:
            if not _backend_seen_up:
                await self._wait_for_backend()
                _backend_seen_up = True
            agent = building = MimirAgent(model=self.model, session_id=self.session_id)
            await agent.connect_servers(all_servers())
            agent.seed_classification_from_caps()

            # Approvals, outside paths and questions become cards on this socket.
            agent.human = self.human
            # Let the loop pick up messages the user types WHILE the agent is
            # working (chat-while-busy steering); drained at each step boundary.
            agent.poll_steer = self._drain_steer_q
            # Let the loop detach a long run: the agent ends its turn and this
            # worker watches the run to completion, then notifies + auto-resumes.
            agent.register_background_job = self.jobs.register
            # The twin of the hook above: what the dispatch guard reads to know a run
            # is already being watched, so asking whether it is done can be refused.
            agent.watched_background_jobs = self.jobs.watched

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
        """Ask the backend to load the model now, off the first query's clock."""

        model = self.model
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: get_backend().prewarm(model))
        except Exception:
            logger.debug("_prewarm_model: a best-effort step failed and was skipped", exc_info=True)
            pass  # Non-fatal — first real query will trigger load.

    async def _query_loop(self) -> None:
        # Bound to this loop, which is the one it is awaited on. A turn queued before
        # it exists is found by the first look below.
        self._query_wake = asyncio.Event()
        while True:
            # Cleared before the look, not after the wait: an item queued between the
            # two then wakes the next wait instead of being missed by it.
            self._query_wake.clear()
            try:
                item = self._query_q.get_nowait()
            except _queue.Empty:
                await self._query_wake.wait()
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
        self.human.begin_turn(resume, item.get("answer"))
        if self._agent is not None:
            self._agent.turn.deferred_prompts = []
            self._agent.turn.deferred_turn = None

        # Clear cancel flag from any previous cancellation before starting.
        if self._agent is not None:
            self._agent.turn.cancel_flag.clear()
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
                self._agent.turn.cancel_flag.clear()
            self.human.end_turn()

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
            full = self._agent.turn.last_full_messages
            answer_ev["_full"] = list(full) if full else None
            answer_ev["_turn_start"] = self._agent.turn.last_turn_start
            # This turn's agent, not whichever one is on screen: several conversations
            # answer at once, and their modes need not agree.
            answer_ev["_context_mode"] = getattr(self._agent, "context_mode", "full")
        answer_ev["_submitted_len"] = getattr(self, "_turn_submitted_len", None)
        deferred = self.human._deferred_record()
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

        agent = self._agent
        if agent is None:
            return []
        snapshots = agent.approvals.batch.snapshots
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
            agent.approvals.batch.note_reviewed(path, current)
            before_lines = original.splitlines(keepends=True) if original is not None else []
            after_lines  = current.splitlines(keepends=True)
            if before_lines == after_lines:
                continue
            rel = os.path.relpath(path)
            diff_lines = list(difflib.unified_diff(
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
            logger.debug("_push_todos: a best-effort step failed and was skipped", exc_info=True)

    def _own_session(self) -> str | None:
        """The conversation this worker works for — never the one on screen.

        It fell back to the session on screen once, for a worker built before any
        session was known; one with a session of its own reading the screen instead
        wrote another conversation's checklist the moment the user looked elsewhere.
        A worker with no session (the pre-agent stand-in) has none.
        """
        return self.session_id

    def _load_todos(self) -> list:
        try:
            return load_todo_items(_todo_file_for_session(self._own_session()))
        except Exception:
            logger.debug("_load_todos: a best-effort step failed and was skipped", exc_info=True)
            return []

    def _clear_todos(self) -> None:
        """Wipe this worker's own session's todo file."""
        try:
            todo_file = _todo_file_for_session(self._own_session())
            if os.path.exists(todo_file):
                open(todo_file, "w").close()
        except Exception:
            logger.debug("_clear_todos: a best-effort step failed and was skipped", exc_info=True)

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

    def set_non_interactive(self, value: bool = True) -> None:
        """Tell the agent there is no terminal behind it.

        Read by the policy ``violations._is_interactive_session``, which otherwise has to
        infer it from the tty — and a detached server's stdout is a log file, which is
        not a tty but also not proof that nobody is reachable. Saying so plainly is
        what keeps the interactive path from being attempted at all.
        """
        agent = self._agent
        if agent is not None:
            agent.non_interactive = bool(value)

    def is_parked(self) -> bool:
        """True while the running turn waits on the user."""
        return self.is_busy() and self.human.awaiting_user()

    def submit_resume(self, record: dict, answer: dict, history: list,
                      session_id: str | None) -> None:
        """Queue the turn that picks a deferred one up with the user's *answer*."""
        self._query_q.put({"text": record.get("query", ""), "history": history,
                           "session_id": session_id, "resume": record,
                           "answer": answer})
        self.human.has_deferral = False   # answered: the turn is nobody's debt any more
        self._wake_query_loop()

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
        msgs = self._agent.turn.last_full_messages
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
        start = self._agent.turn.last_turn_start
        return start if isinstance(start, int) else None

    def live_history(self) -> list | None:
        """The in-flight transcript of the query currently running (system excluded).

        The agent mutates this list in place at every step, so reading it gives the
        context as it grows *during* a turn — which is what the context bar needs.
        ``None`` when no query is running.
        """
        if self._agent is None:
            return None
        msgs = self._agent.turn.live_messages
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
        self._wake_query_loop()

    def _wake_query_loop(self) -> None:
        """Wake the query loop, from whichever thread queued something for it."""
        loop, wake = self._loop, self._query_wake
        if loop is None or wake is None:
            return   # not started yet: its first look finds what was queued
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            pass     # the loop has closed; nothing is left to wake

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
        self.human.flush()
        self._drain_steer_q()


    def _drain_steer_q(self) -> list[str]:
        """Pop and return all queued steer messages (the agent's ``poll_steer``)."""
        out: list[str] = []
        while True:
            try:
                out.append(self._steer_q.get_nowait())
            except _queue.Empty:
                break
        return out

    def apply_setting(self, name: str, value: Any) -> str:
        """Apply one of the user's settings to this agent: "" on success, else why not.

        The caller reports a refusal to the front-end — swallowed, it left the webview
        showing a mode the agent was never switched to. With no agent yet there is
        nothing to apply: the pool records the setting and replays it on the agent
        built later. Called from the WS event loop while the worker thread may be
        running a query; a setting only rebinds an attribute the loop reads afresh at
        each step, so it applies from the next step on.
        """
        if self._agent is None:
            return ""
        try:
            self._agent.apply_setting(name, value)
        except ValueError as exc:
            return str(exc)
        if name == "model":
            self.model = value
        return ""

    def setting(self, name: str, default: Any = None) -> Any:
        """This agent's *name* setting, or *default* while there is no agent.

        The default is the caller's to give, and it matters: the context budget is
        derived from ``context_mode``, and answering "compact" for a conversation that
        was running in full mode sized its window at 32k.
        """
        if self._agent is None:
            return read_setting(None, name) if default is None else default
        return read_setting(self._agent, name)

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
            return [m for m in get_backend().served_models() if isinstance(m, str) and m.strip()]
        except Exception:
            logger.debug("served_models: a best-effort step failed and was skipped", exc_info=True)
            return []

    def get_temperature_state(self) -> dict:
        """Whether this backend honours a temperature, and the one set for the model.

        ``value`` None is the model's own. Before the agent exists (the greeting is
        sent while the backend is still coming up) it is read from the stored
        preference, which is exactly what the agent will load.
        """
        backend = os.environ.get("LLM_BACKEND", "vllm").lower()
        if self._agent is not None:
            value = getattr(self._agent, "temperature", None)
        else:
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

    def set_enabled(self, kind: str, name: str, enabled: bool) -> None:
        if self._agent is not None:
            self._agent.set_enabled(kind, name, enabled)

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
        """Call *tool* directly (``call_tool_directly``) on the worker's OWN loop.

        Returns a Future of the decoded payload. Same constraint as
        :meth:`resolve_resources`: the MCP ClientSession objects are bound to this loop,
        so a call made from the WebSocket thread is scheduled onto it rather than
        awaited where it was asked for.
        """
        if self._agent is None or self._loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result(err(NO_AGENT))
            return fut
        return asyncio.run_coroutine_threadsafe(
            call_tool_directly(self._agent, tool, args), self._loop)

    def panel_sections(self, *, force: bool = False) -> Any:
        """Collect every section the connected servers can fill. Future of a list.

        Asked by capability, never by name: a server declares ``PANEL_REPORT`` and the
        section it fills, and a plugin server's section appears with no client edit.
        Each answer is whatever that server chose to show — the panel renders lines, it
        does not interpret them.

        *force* is the reader asking for a re-read rather than a timer asking for the
        current picture. A section that costs an ``sinfo`` or a reach across ``ssh`` is
        served from the owning server's cache, which is what makes a panel on a timer
        affordable and what made pressing ⟳ hand back the page just read. A server that
        caches names the argument meaning "read it again" in its ``panel`` descriptor;
        this sets it, and only for that press.

        Straight to the owning session, like the slash commands above: this is the
        user opening a drawer, not the model taking a step, and routing it through the
        guardrail pipeline would scatter tool cards through their transcript.
        """

        if self._agent is None or self._loop is None:
            fut: Any = concurrent.futures.Future()
            fut.set_result([])
            return fut
        agent = self._agent
        declared = _declared(getattr(agent, "tool_caps", None))

        async def _ask(tool: str, spec: dict) -> dict | None:
            owner = (getattr(agent, "tool_owner", None) or {}).get(tool)
            if owner is None:
                return None
            args = dict(spec.get("args") or {})
            if force and spec.get("refresh_arg"):
                args[str(spec["refresh_arg"])] = True
            try:
                raw = await agent.sessions[owner].call_tool(tool, args)
                text = normalize_tool_content(raw)
                return json.loads(text) if isinstance(text, str) else (text or {})
            except Exception as exc:
                # One server that cannot answer costs its own section, never the
                # panel: the others have already said something worth showing.
                logger.debug("panel section %r failed: %s", tool, exc)
                return None

        async def _run() -> list[dict]:
            # All at once: the sections are independent, they are owned by different
            # processes, and the slow one is slow because it is waiting on a scheduler.
            # Asked in turn, the drawer cost the sum of every server's wait; asked
            # together it costs the longest. The order below is still the declared one.
            payloads = await asyncio.gather(
                *(_ask(tool, spec) for tool, spec in declared))
            out: list[dict] = []
            for (tool, spec), payload in zip(declared, payloads):
                if payload is None:
                    continue
                lines = payload.get("lines") or []
                detail = payload.get("detail") or ""
                # A nested half, for a server whose answer is a tree rather than a
                # list: a scheduler, the kinds of node on it, what one of them is.
                # Passed through unread, exactly as the lines are — this end renders
                # what a server chose to say and never decides what a field means.
                groups = payload.get("groups") or []
                if not lines and not detail and not groups:
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
                    "groups": groups,
                    "detail": detail,
                })
            return out

        return asyncio.run_coroutine_threadsafe(_run(), self._loop)

    def watched_job_keys(self) -> list[str]:
        """Keys of the runs this conversation still has in flight (``JobWatcher``)."""
        return self.jobs.watched_job_keys()

    def watched_runs(self) -> list[dict]:
        """What is running outside the current turn, for the panel (``JobWatcher``)."""
        return self.jobs.watched_runs()

    def rearm_detached_jobs(self) -> dict:
        """Put watchers back on this session's runs found on disk (``JobWatcher``)."""
        return self.jobs.rearm_detached_jobs()

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
            return get_backend().count_text_tokens(self.model, text, allow_network=False)
        except Exception:
            logger.debug("_count_tokens: a best-effort step failed and was skipped", exc_info=True)
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
            logger.debug("context_overhead_tokens: a best-effort step failed and was skipped", exc_info=True)
            return 0

    def cancel(self) -> bool:
        """Cancel the currently running query task. Returns True if a task was cancelled."""
        # Set the cancel flag first so _stream_chat aborts mid-chunk immediately,
        # without waiting for the next asyncio await point.
        if self._agent is not None:
            self._agent.turn.cancel_flag.set()
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
        for task in self.jobs.tasks():
            if self._loop is not None and not task.done():
                self._loop.call_soon_threadsafe(task.cancel)
        self._query_q.put(None)
        self._wake_query_loop()

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

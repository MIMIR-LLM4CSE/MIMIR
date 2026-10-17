"""The background agent worker for the WebSocket server.

``AgentWorker`` runs MimirAgent in a dedicated thread with its own asyncio event
loop so blocking approval prompts never freeze the WebSocket event loop. It talks
to the connections exclusively through thread-safe queues. The wire protocol is
documented in ``server.main``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import queue as _queue
import threading
from typing import Any

from .....servers._shared.responses import err
from ....agent_core import MimirAgent
from ....config import TEMPERATURE_BACKENDS
from ....config.models import thinking_can_disable, thinking_profile
from ....config.preferences import load_temperature
from ....core.settings import read_setting
from ....extensions import all_servers
from ....query_engine.backends.factory import get_backend
from ....tool_execution.direct import NO_AGENT, call_tool_directly

# Import the shared runtime FIRST so its cwd bootstrap runs before config.paths
# (and the backend factory) capture the workspace root at import time.
from ..runtime import (
    ROUTER,
)

# After the runtime: it reads the workspace root that the bootstrap above pins.
from .human import HumanBridge
from .jobs import JobWatcher
from .backend_wait import WorkerBackendWait
from .outbox import Outbox
from .review import WorkerReview
from .panel import WorkerPanel
from .usage import WorkerUsage

logger = logging.getLogger(__name__)

# The backend is the server's, not a conversation's: once one agent has seen it answer,
# the next conversation's agent does not wait on it again.
_backend_seen_up = False


class AgentWorker(WorkerUsage, WorkerPanel, WorkerReview, WorkerBackendWait):
    """Runs MimirAgent in a dedicated background thread with its own event loop.

    Thread-safe queues carry events to the WS layer and approval responses back.
    """

    _thread: threading.Thread | None

    @classmethod
    def detached(cls, model: str) -> "AgentWorker":
        """The worker of a conversation that has no agent yet (:class:`NoAgentWorker`)."""
        return NoAgentWorker(model)

    def __init__(self, model: str, session_id: str | None = None, *,
                 start: bool = True) -> None:
        """*start* False builds the fields only: no thread, no servers, no agent."""
        self._init_fields(model, session_id)
        if not start:
            return
        thread = threading.Thread(target=self._main, name="mimir-worker", daemon=True)
        self._thread = thread
        thread.start()
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
        """Every field, and nothing that starts running (see :class:`NoAgentWorker`)."""
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
        self.out_q = Outbox()                               # agent → WS
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
        self._thread = None

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
        ROUTER.register(tid, lambda text: self.out_q.put({"type": "output", "text": text}))

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
            ROUTER.unregister(tid)
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
        ``process_response`` in the agent loop). Returns ``None`` when no query has
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
        uses. Reporting it from here keeps the panel's model picker working when the
        extension host's own probe is swallowed by a proxy.

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


    def watched_job_keys(self) -> list[str]:
        """Keys of the runs this conversation still has in flight (``JobWatcher``)."""
        return self.jobs.watched_job_keys()

    def watched_runs(self) -> list[dict]:
        """What is running outside the current turn, for the panel (``JobWatcher``)."""
        return self.jobs.watched_runs()

    def rearm_detached_jobs(self) -> dict:
        """Put watchers back on this session's runs found on disk (``JobWatcher``)."""
        return self.jobs.rearm_detached_jobs()


    def cancel(self) -> bool:
        """Cancel the currently running query task. Returns True if a task was cancelled."""
        # Set the cancel flag first so stream_chat aborts mid-chunk immediately,
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


class NoAgentWorker(AgentWorker):
    """A conversation's worker before its first query: no thread, no servers, no agent.

    Every read of ``AgentWorker`` answers for an absent agent (``setting`` gives the
    caller's default, ``toggles_state`` is empty, the temperature is the stored
    preference) and every setter does nothing, so this class only refuses what needs
    an agent: a query. The pool builds a real worker for that.
    """

    def __init__(self, model: str) -> None:
        super().__init__(model, start=False)

    def submit_query(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("a conversation's query goes to the worker the pool builds for it")

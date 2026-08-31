"""
MCP Spawn-Agent Server
======================
Lets the orchestrator delegate a sub-task to a fresh MimirAgent instance that
runs to completion and returns its answer as a string.  Multiple spawn_agent
calls emitted in a single model step run **concurrently**: the parent gathers
them, and this tool is async so a running child never blocks this server's loop
(a sync tool is awaited inline by FastMCP, which would serialize the fan-out).

While a child runs, its own tool calls are streamed back to the caller as MCP
progress notifications, so the UI can show what the child is doing instead of one
row spinning for the whole cap.

Tools:
  1. spawn_agent(task, context?, role?, max_steps?)
        — spin up a child MimirAgent, run it, return its answer.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
from concurrent.futures import Future
from typing import Any, Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from capabilities import DELEGATE, PLAN_READONLY, REVERSIBLE, tool_caps
from mcp.server.fastmcp import Context, FastMCP
from responses import err, ok

mcp = FastMCP("spawn_agent")

# The two roles a child can take. EXPLORE is the read-only one: it runs in a
# read-only mode, so what keeps it from writing is the client's capability filter
# and dual-use call gate, not a hand-kept list of servers.
ROLE_EXPLORE = "explore"
ROLE_TASK = "task"

# The read-only mode an explorer runs in (one of client config.models.READONLY_MODES).
# "ask" rather than "plan": the child answers a question, it does not draft a plan.
_READONLY_CHILD_MODE = "ask"

# Wall for one sub-agent run, enforced here. What the tool DECLARES to the dispatcher
# is deliberately larger, so this cap fires first and hands back the partial answer
# instead of the parent killing the call with nothing to show.
SUBAGENT_HARD_CAP_SECS = 600

# What an explorer owes back. Its own mode prompt already asks for cited prose; this
# says the part that is about the *parent*: a conclusion costs the caller a paragraph
# of context, the file contents it read would cost the window they were meant to save.
_EXPLORE_BRIEF = (
    "You are an exploration sub-agent. Answer the question below by reading the code, "
    "and return a CONCLUSION: what you found, with the concrete file paths, symbols and "
    "line numbers that back each claim. Do not paste the file contents you read — the "
    "caller wants your finding, not your reading. Say plainly what you could not "
    "establish rather than guessing. You own the breadth of the sweep: search widely "
    "enough to be sure, then stop."
)

# ── Sub-agent output routing ──────────────────────────────────────────────────
# This server is a subprocess whose stdout IS the JSON-RPC pipe, so anything
# printed here corrupts the protocol. Two leaks are plugged: the child agent gets
# an event sink (without one, event_sink.emit falls back to printing every event),
# and stdout is pointed at stderr for whatever still prints.

_CHILD_QUEUE_MAX = 256      # child events buffered between two forwarding ticks
_MAX_EVENTS_PER_TICK = 8    # forwarded per tick, so a tool storm cannot hog the loop
_MAX_EVENTS_PER_RUN = 500   # ceiling per child run; past it only the trailer is sent
_POLL_SECS = 0.05

_stdout_silenced = False


def _silence_stdout_once() -> None:
    """Point stdout at stderr, once, on first use — never at import time.

    The stdio transport wraps ``sys.stdout.buffer`` once at startup; rebinding
    before that would make it wrap *stderr* and the server would go mute. Doing it
    afterwards is harmless (the transport holds its own reference), and it is never
    restored: two concurrent children would restore each other's saved value.
    """
    global _stdout_silenced
    if _stdout_silenced:
        return
    sys.stdout = sys.stderr
    _stdout_silenced = True


def _clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _compact_event(ev: dict) -> dict | None:
    """One child engine event in wire form, or None when it is not worth forwarding.

    Only the child's tool activity travels: tokens, thinking and diffs would cost
    far more than they show, and the caller renders these as ordinary tool rows.
    Keys are short because each event rides in a progress notification.
    """
    kind = ev.get("type")
    if kind == "tool_call":
        return {
            "v": 1, "t": "tc",
            "i": str(ev.get("id") or ""),
            "n": str(ev.get("name") or ""),
            "l": _clip(ev.get("label"), 120),
            "d": _clip(ev.get("detail"), 160),
        }
    if kind == "tool_result":
        out = {
            "v": 1, "t": "tr",
            "i": str(ev.get("id") or ""),
            "ok": bool(ev.get("ok")),
            "s": _clip(ev.get("summary"), 160),
        }
        ms = ev.get("duration_ms")
        if isinstance(ms, (int, float)):
            out["ms"] = int(ms)
        return out
    return None


def _make_child_sink(q: queue.Queue, counters: dict) -> Callable[[dict], None]:
    """The child's event_callback: enqueue, never block, never raise.

    Raising here would put the child's engine back on emit()'s print fallback —
    straight into the JSON-RPC pipe — so a full queue drops the event and counts it.
    """
    def _sink(ev: dict) -> None:
        try:
            compact = _compact_event(ev)
            if compact is None:
                return
            q.put_nowait(compact)
        except Exception:
            counters["dropped"] = counters.get("dropped", 0) + 1
    return _sink


async def _report(ctx: Context | None, state: dict, payload: dict) -> None:
    """Send one event to the caller. A dead pipe must not fail the child's run."""
    if ctx is None:
        return
    state["sent"] = state.get("sent", 0) + 1
    try:
        await ctx.report_progress(
            progress=state["sent"],
            message=json.dumps(payload, separators=(",", ":")),
        )
    except Exception:
        pass


async def _forward_pending(ctx: Context | None, q: queue.Queue, state: dict) -> None:
    """Drain a bounded slice of the child's events to the caller."""
    for _ in range(_MAX_EVENTS_PER_TICK):
        try:
            ev = q.get_nowait()
        except queue.Empty:
            return
        if state.get("sent", 0) >= _MAX_EVENTS_PER_RUN:
            state["dropped"] = state.get("dropped", 0) + 1
            continue
        await _report(ctx, state, ev)


# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool(**tool_caps(
    caps=[DELEGATE, PLAN_READONLY],
    # Reversible, hence not approval-gated: a card in front of every exploration is a
    # card in front of the behaviour this tool exists to make cheap. A writing child
    # is gated by its own approval layer, and rejected outright in a read-only mode.
    reversibility=REVERSIBLE,
    # Deliberately above the tool's own SUBAGENT_HARD_CAP_SECS, so the inner cap fires
    # first and returns the child's partial answer instead of the dispatcher killing
    # the call with nothing to show.
    timeout_secs=SUBAGENT_HARD_CAP_SECS + 60,
    # Dual-use: in a read-only mode only the exploring role may run. Declared here so
    # the client's call-time gate reads it off the descriptor.
    readonly_when={"arg": "role", "values": [ROLE_EXPLORE]},
    label="Sub-agent ({role}): {task}",
    read_only=False,
))
async def spawn_agent(
    task: str,
    context: str = "",
    role: str = ROLE_EXPLORE,
    max_steps: int = 30,
    ctx: Context | None = None,
) -> dict:
    """Hand a self-contained sub-task to a fresh sub-agent, and get its answer back.

    The sub-agent has its own conversation history and execution context, so what it
    reads never enters yours — only its answer does. Emit several calls **in the same
    response** and they run concurrently; that fan-out is the point of the tool, and a
    broad sweep splits naturally into independent questions.

    Args:
        task:       The complete, self-contained task or question for the sub-agent.
                    It sees none of your conversation, so say everything it needs.
        context:    Optional extra context (findings so far, constraints) prepended
                    to the task.
        role:       "explore" (default) — read-only reconnaissance: the child cannot
                    write, execute or mutate anything, and answers in prose citing the
                    files and symbols it found. "task" — a child with the full
                    workspace toolkit, for a separable piece of work that must write.
        max_steps:  Max tool-call steps the sub-agent may take (default 30).
    Returns:
        On success (the sub-agent ran to a result):
            {"status": "ok", "answer": "…", "completed": bool, "files_read": [...],
             "files_written": [...]}
            - ``answer``        — the sub-agent's final answer (primary payload).
            - ``completed``     — False when the sub-agent ran out of steps or reported
                                  the task incomplete (the answer is still informative).
            - ``files_read``    — what the child actually opened, so the evidence it
                                  gathered is on the record for the caller too.
            - ``files_written`` — workspace files the sub-agent modified (always empty
                                  for "explore"); lets a parent coordinating concurrent
                                  sub-agents detect overlapping edits.
        On failure (the sub-agent crashed or hit the hard time cap):
            {"status": "error", "error": "…", "answer": "<partial>", ...}
            — distinct ``status`` so the orchestrator can branch on failure without
              string-matching the answer text.
    """
    role = (role or ROLE_EXPLORE).strip().lower()
    if role not in (ROLE_EXPLORE, ROLE_TASK):
        return err(
            f"unknown role {role!r}: expected {ROLE_EXPLORE!r} (read-only reconnaissance) "
            f"or {ROLE_TASK!r} (full workspace toolkit)",
            answer="", completed=False, files_read=[], files_written=[],
        )

    _silence_stdout_once()

    # The child runs in a dedicated thread with its own event loop: its MCP exit
    # stack must be opened and closed in one task on one loop, and the hard cap
    # below must not become a task cancellation in the middle of a stdio_client.
    # This tool stays async so waiting on that thread never blocks this server's
    # loop — that is what makes a fan-out of several calls actually concurrent.
    events: queue.Queue = queue.Queue(maxsize=_CHILD_QUEUE_MAX)
    counters: dict = {"dropped": 0}
    state: dict = {"sent": 0, "dropped": 0}
    future: Future[dict] = Future()

    def _thread_main() -> None:
        try:
            result = asyncio.run(_run_sub_agent(
                task, context, role, max_steps,
                on_event=_make_child_sink(events, counters),
            ))
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()

    loop = asyncio.get_running_loop()
    deadline = loop.time() + SUBAGENT_HARD_CAP_SECS
    while not future.done() and loop.time() < deadline:
        await _forward_pending(ctx, events, state)
        await asyncio.sleep(_POLL_SECS)
    # One last drain, before the timeout branch too: the child's final tool result
    # is what tells the caller where a run that ran out of time actually stopped.
    await _forward_pending(ctx, events, state)
    dropped = counters.get("dropped", 0) + state.get("dropped", 0)
    if dropped:
        await _report(ctx, state, {"v": 1, "t": "end", "dropped": dropped})

    if not future.done():
        return err(
            f"sub-agent failed: exceeded the {SUBAGENT_HARD_CAP_SECS}s cap",
            answer="", completed=False, files_read=[], files_written=[],
        )
    try:
        result = future.result(timeout=0)
    except Exception as exc:
        # The child's thread crashed — a genuine failure, not an "ok" result.
        return err(
            f"sub-agent failed: {exc}",
            answer="", completed=False, files_read=[], files_written=[],
        )

    if result.get("error"):
        # The sub-agent's own run() raised — surface as an error, keep the partial answer.
        return err(
            f"sub-agent crashed: {result['error']}",
            answer=result.get("answer", ""),
            completed=False,
            files_read=result.get("files_read", []),
            files_written=result.get("files_written", []),
        )
    if not str(result.get("answer") or "").strip():
        # The answer IS the payload — a blank one carries nothing back, whatever the
        # child did. Reported "ok"/completed, it read as a sub-agent that ran and found
        # nothing to say, and the parent dropped delegation for the rest of the run.
        return err(
            "sub-agent returned no answer: nothing was delegated back. Re-issue the "
            "call with a narrower, self-contained question, or do the work yourself.",
            answer="",
            completed=False,
            files_read=result.get("files_read", []),
            files_written=result.get("files_written", []),
        )
    return ok({
        "answer": result["answer"],
        "completed": result["completed"],
        "files_read": result["files_read"],
        "files_written": result["files_written"],
    })


async def _run_sub_agent(
    task: str,
    context: str,
    role: str,
    max_steps: int,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Async implementation: create MimirAgent, wire tools, run query.

    Returns ``{"answer", "completed", "files_read", "files_written", "error"}``;
    ``error`` is None on a clean run (the task may still be incomplete — see
    ``completed``).
    """
    try:
        from mimir.client.agent_core import MimirAgent
        from mimir.client.config import DEFAULT_MODEL
    except ImportError:
        # Running as a subprocess MCP server — adjust sys.path.
        _root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from mimir.client.agent_core import MimirAgent
        from mimir.client.config import DEFAULT_MODEL

    exploring = role == ROLE_EXPLORE

    # Always use the same model as the parent process — the backend (Ollama or
    # vLLM) only serves one model at a time, so using a different one would fail.
    # The model is read from MIMIR_DEFAULT_MODEL (set by the parent before
    # spawning servers) or from the config DEFAULT_MODEL.
    _model = os.environ.get("MIMIR_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL
    agent = MimirAgent(model=_model)
    try:
        return await _drive_sub_agent(
            agent, task, context, role, exploring, max_steps, on_event)
    finally:
        # Close the child's MCP stdio sessions HERE, in the very task that opened
        # them. Left to asyncio.run's shutdown_asyncgens, each stdio_client would be
        # closed from another task and blow up on anyio's cancel-scope check (and
        # leak one server subprocess per connected server).
        try:
            await agent.cleanup()
        except Exception as exc:  # teardown races must not mask the child's answer
            print(f"spawn_agent: sub-agent cleanup warning: {exc}", file=sys.stderr)


async def _drive_sub_agent(
    agent,
    task: str,
    context: str,
    role: str,
    exploring: bool,
    max_steps: int,
    on_event: Callable[[dict], None] | None = None,
) -> dict:
    """Configure the child agent, run it, and report what it did."""
    # _run_sub_agent already fixed sys.path if needed, so a plain import is safe here.
    from mimir.client.extensions import all_servers
    from mimir.client.context.capabilities import explorer_servers

    # Sub-agents don't reason out loud: set the rung, not just the run() flag, since
    # the loop re-reads the agent's depth every step (the user steering the parent's
    # thinking must not leak into a child run).
    agent.set_thinking_depth(0)
    # An explorer is read-only because of the MODE it runs in, not because of which
    # servers it connected: the mode strips every plan-blocked tool and gates the
    # dual-use shell at call time. Set on the agent as well as passed to run(), so the
    # loop's mode tracking starts where it ends and reads no switch on the first step.
    if exploring:
        agent.set_mode(_READONLY_CHILD_MODE)

    _servers = all_servers()
    servers_to_connect = (
        # A cost filter, not the guarantee: starting every MCP server for a
        # reconnaissance run is what it saves.
        {k: v for k, v in _servers.items() if k in explorer_servers()}
        if exploring
        else {k: v for k, v in _servers.items() if k != "agent"}  # avoid recursive spawn
    )

    for name, script in servers_to_connect.items():
        try:
            await agent.connect_server(name, script)
        except Exception as exc:
            print(f"spawn_agent: could not connect server '{name}': {exc}", file=sys.stderr)

    # Seed classification from this sub-agent's own (subset) registry — keeps
    # approval/plan-block/caching correct and per-agent (the sub-agent connects
    # fewer servers than the parent).
    agent.seed_classification_from_caps()

    # The sub-agent operates without a session sidecar (CLI mode), so it never
    # touches the parent's todo list.
    agent._spawn_mode = True  # marker flag (not currently used by loop)

    brief = _EXPLORE_BRIEF if exploring else ""
    query = "\n\n---\n\n".join(p for p in (brief, context.rstrip(), task) if p)

    print(f"⟳ sub-agent starting ({role}): {task[:80]}{'…' if len(task) > 80 else ''}",
          file=sys.stderr)
    error: str | None = None
    try:
        answer = await agent.run(
            query=query,
            max_steps=max_steps,
            mode=_READONLY_CHILD_MODE if exploring else "agent",
            streaming=False,   # sub-agents don't stream tokens to the UI
            thinking=False,
            # Binding a sink does double duty: the caller gets to see what the child
            # is doing, and the child's engine stops falling back to printing every
            # event — which here means printing into the JSON-RPC pipe.
            event_callback=on_event,
        )
    except Exception as exc:
        answer = f"[sub-agent error] {exc}"
        error = str(exc)
    print(f"✓ sub-agent finished: {task[:60]}{'…' if len(task) > 60 else ''}",
          file=sys.stderr)

    # What the child touched this run (recorded by _update_carry_context). `files_read`
    # goes back so the caller can record the evidence a delegated sweep produced.
    _carry = agent._carry_context
    files_read = sorted(_carry.get("read_files", set()))
    files_written = sorted(_carry.get("last_query_written_files", set()))
    # "completed" = the task itself finished; distinct from the run succeeding.
    # finalize_incomplete_answer owns its headlines (is_incomplete_answer matches
    # them); the hard step limit yields "Reached the maximum number of steps…".
    from mimir.client.guardrails.workflow import is_incomplete_answer

    completed = error is None and not (
        not answer.strip()
        or is_incomplete_answer(answer)
        or answer.startswith("Reached the maximum number of steps")
    )
    return {
        "answer": answer,
        "completed": completed,
        "files_read": files_read,
        "files_written": files_written,
        "error": error,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

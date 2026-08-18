"""
MCP Spawn-Agent Server
======================
Lets the orchestrator delegate a sub-task to a fresh MimirAgent instance that
runs to completion and returns its answer as a string.  Multiple spawn_agent
calls emitted in a single model step are executed **concurrently** by the
existing asyncio.gather dispatch in agent_loop.py.

Tools:
  1. spawn_agent(task, context?, model?, max_steps?, readonly?)
        — spin up a child MimirAgent, run it, return its answer.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from concurrent.futures import Future

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from responses import err, ok

mcp = FastMCP("spawn_agent")

# ── Sub-agent output routing ──────────────────────────────────────────────────
# Each sub-agent runs in the *same* background thread as the parent tool call,
# so stdout goes through the parent's _ROUTER registration.  We prefix every
# line with the task name to make it identifiable in the UI.

def _make_prefix_writer(prefix: str, original_write):
    def _write(text: str) -> None:
        if text.strip():
            original_write(f"[{prefix}] {text}")
        else:
            original_write(text)
    return _write


# ── Tool ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def spawn_agent(
    task: str,
    context: str = "",
    model: str = "",
    max_steps: int = 30,
    readonly: bool = False,
) -> dict:
    """Spawn a fresh sub-agent to handle a self-contained sub-task.

    The sub-agent gets its own conversation history, execution context, and
    (if readonly=False) access to all workspace tools.  Multiple spawn_agent
    calls within a single model step run **concurrently**.

    Args:
        task:       The complete task description for the sub-agent.
        context:    Optional extra context (file snippets, prior findings, …)
                    to prepend to the task prompt.
        model:      Override the LLM model (defaults to the value of
                    MIMIR_DEFAULT_MODEL env var, then the parent's active model).
        max_steps:  Max tool-call steps the sub-agent may take (default 30).
        readonly:   If True, the sub-agent is connected only to read/search
                    tools — it cannot write files or run code.
    Returns:
        On success (the sub-agent ran to a result):
            {"status": "ok", "answer": "…", "completed": bool, "files_written": [...]}
            - ``answer``        — the sub-agent's final answer string (primary payload).
            - ``completed``     — False when the sub-agent ran out of steps or reported
                                  the task incomplete (the answer is still informative).
            - ``files_written`` — workspace files the sub-agent modified (empty when
                                  readonly); lets a parent coordinating concurrent
                                  sub-agents detect overlapping edits.
        On failure (the sub-agent crashed or hit the 10-minute hard cap):
            {"status": "error", "error": "…", "answer": "<partial>", ...}
            — distinct ``status`` so the orchestrator can branch on failure without
              string-matching the answer text.
    """
    # Always run in a dedicated thread with its own event loop so this
    # synchronous MCP tool is safe to call from inside a running asyncio loop.
    future: Future[dict] = Future()

    def _thread_main() -> None:
        try:
            result = asyncio.run(_run_sub_agent(task, context, model, max_steps, readonly))
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)

    t = threading.Thread(target=_thread_main, daemon=True)
    t.start()
    try:
        result = future.result(timeout=600)  # 10-minute hard cap per sub-agent
    except Exception as exc:
        # Thread crash or hard-cap timeout — a genuine failure, not an "ok" result.
        return err(f"sub-agent failed: {exc}", answer="", completed=False, files_written=[])

    if result.get("error"):
        # The sub-agent's own run() raised — surface as an error, keep the partial answer.
        return err(
            f"sub-agent crashed: {result['error']}",
            answer=result.get("answer", ""),
            completed=False,
            files_written=result.get("files_written", []),
        )
    return ok({
        "answer": result["answer"],
        "completed": result["completed"],
        "files_written": result["files_written"],
    })


async def _run_sub_agent(
    task: str,
    context: str,
    model_override: str,
    max_steps: int,
    readonly: bool,
) -> dict:
    """Async implementation: create MimirAgent, wire tools, run query.

    Returns ``{"answer", "completed", "files_written", "error"}``; ``error`` is None on
    a clean run (the task may still be incomplete — see ``completed``).
    """
    try:
        from mimir.client.agent_core import MimirAgent
        from mimir.client.config import DEFAULT_MODEL
        from mimir.client.extensions import all_servers
        from mimir.client.context.capabilities import readonly_servers
    except ImportError:
        # Running as a subprocess MCP server — adjust sys.path.
        _root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        )
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from mimir.client.agent_core import MimirAgent
        from mimir.client.config import DEFAULT_MODEL
        from mimir.client.extensions import all_servers
        from mimir.client.context.capabilities import readonly_servers

    # Read-only server subset: search/read tools only, no mutations.
    _READONLY_SERVERS = readonly_servers()

    # Always use the same model as the parent process — the backend (Ollama or
    # vLLM) only serves one model at a time, so using a different one would fail.
    # The model is read from MIMIR_DEFAULT_MODEL (set by the parent before
    # spawning servers) or from the config DEFAULT_MODEL.  The model_override
    # arg is intentionally ignored.
    _model = os.environ.get("MIMIR_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL
    agent = MimirAgent(model=_model)
    # Sub-agents don't reason out loud: set the rung, not just the run() flag, since
    # the loop re-reads the agent's depth every step (the user steering the parent's
    # thinking must not leak into a child run).
    agent.set_thinking_depth(0)

    _servers = all_servers()
    servers_to_connect = (
        {k: v for k, v in _servers.items() if k in _READONLY_SERVERS}
        if readonly
        else {k: v for k, v in _servers.items() if k != "agent"}  # avoid recursive spawn
    )

    for name, script in servers_to_connect.items():
        try:
            await agent.connect_server(name, script)
        except Exception as exc:
            print(f"spawn_agent: could not connect server '{name}': {exc}")

    # Seed classification from this sub-agent's own (subset) registry — keeps
    # approval/plan-block/caching correct and per-agent (the sub-agent connects
    # fewer servers than the parent).
    agent.seed_classification_from_caps()

    # Give the sub-agent its own session context so it doesn't touch the
    # parent's todo_list.md.  An empty active_session means it uses the
    # legacy shared path — we override via environment is not ideal; instead
    # we just ensure it uses no active session by writing nothing.
    # The sub-agent operates without a session sidecar (CLI mode).
    agent._spawn_mode = True  # marker flag (not currently used by loop)

    query = task if not context else f"{context.rstrip()}\n\n---\n\n{task}"

    print(f"⟳ sub-agent starting: {task[:80]}{'…' if len(task) > 80 else ''}")
    error: str | None = None
    try:
        answer = await agent.run(
            query=query,
            max_steps=max_steps,
            streaming=False,   # sub-agents don't stream tokens to the UI
            thinking=False,
        )
    except Exception as exc:
        answer = f"[sub-agent error] {exc}"
        error = str(exc)
    print(f"✓ sub-agent finished: {task[:60]}{'…' if len(task) > 60 else ''}")

    # Files the sub-agent modified this run (recorded by _update_carry_context).
    files_written = sorted(agent._carry_context.get("last_query_written_files", set()))
    # "completed" = the task itself finished; distinct from the run succeeding.
    # finalize_incomplete_answer owns its headlines (is_incomplete_answer matches
    # them); the hard step limit yields "Reached the maximum number of steps…".
    from mimir.client.guardrails.workflow import is_incomplete_answer

    completed = error is None and not (
        is_incomplete_answer(answer)
        or answer.startswith("Reached the maximum number of steps")
    )
    return {
        "answer": answer,
        "completed": completed,
        "files_written": files_written,
        "error": error,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")

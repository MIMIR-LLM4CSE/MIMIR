"""Background-job handling for the agent loop.

A tool that launches a long detached run returns a ``background_job`` descriptor
(shape-driven, no tool name hard-coded). With a front-end watcher hook the loop
registers it and tells the model to yield; without one (CLI) it awaits the run
in-turn with ``asyncio.sleep`` (zero model calls). Also emits the ``open_editor``
UI event when a result opts in. Extracted from ``agent_loop.py``.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ..event_sink import emit
from ..context.capabilities import BACKGROUNDABLE, has_cap


def _maybe_emit_open_editor(result_text: str) -> None:
    """Emit an ``open_editor`` UI event when a tool result asks for a file to be
    opened in the editor.

    Shape-driven (like the exec preview): a result payload carrying
    ``open_in_editor: true`` plus an absolute ``path`` opts in — the loop does not
    hard-code any tool name. Used so a written plan (.md) pops open in VS Code for
    the user to read. Best-effort: any parse failure is silently ignored.
    """
    try:
        payload = json.loads(result_text)
    except (TypeError, ValueError):
        return
    if not isinstance(payload, dict) or not payload.get("open_in_editor"):
        return
    path = payload.get("path")
    if isinstance(path, str) and path.strip():
        emit({"type": "open_editor", "path": path})


def _detect_background_job(name: str, result_text: str, agent: Any) -> dict | None:
    """Return a ``background_job`` descriptor from a BACKGROUNDABLE tool result, else None.

    Shape-driven like ``_maybe_emit_open_editor`` — no tool name is hard-coded.
    """
    if not has_cap(name, BACKGROUNDABLE, agent.tool_caps):
        return None
    try:
        payload = json.loads(result_text)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    descriptor = payload.get("background_job")
    return descriptor if isinstance(descriptor, dict) else None


def _maybe_register_background_job(name: str, result_text: str, agent: Any) -> str:
    """Register a completion watcher for a detached run and tell the model to yield.

    Registration is delegated to the optional front-end hook
    ``agent._register_background_job`` (the WebSocket worker sets it; CLI does not),
    mirroring the ``_poll_steer``/``_cancel_flag`` optional-hook pattern.

    Returns the result augmented with a "you were backgrounded, end your turn" note
    **only when a watcher was actually registered**. Callers use ``_detect_background_job``
    to route the CLI (no hook) to ``_await_background_job`` instead. Best-effort.
    """
    descriptor = _detect_background_job(name, result_text, agent)
    if descriptor is None:
        return result_text
    hook = getattr(agent, "_register_background_job", None)
    if not hook:
        return result_text  # no watcher — caller handles the CLI await path
    try:
        registered = bool(hook(descriptor))
    except Exception:
        return result_text
    if not registered:
        return result_text
    job_key = descriptor.get("job_key", "?")
    note = (
        f"\n\n[background] This run is now tracked as background job '{job_key}'. "
        "Do NOT poll its status — end your turn (or start other work / answer the "
        "user). You will be automatically resumed with the results when it completes."
    )
    return result_text + note


async def _await_background_job(descriptor: dict, agent: Any, result_text: str) -> str:
    """Efficiently wait out a detached run in-turn, then append its results.

    The CLI has no persistent worker loop to host a watcher, so instead of the agent
    burning a model call per status poll, we poll the descriptor's read-only status op
    here with ``asyncio.sleep`` (zero model calls) until the run is terminal, fetch the
    summary, and fold it into the tool result. The launch tool already returned, so
    this runs outside the per-tool timeout. Best-effort: failures return what we have.
    """
    status_op   = descriptor.get("status_op") or {}
    summary_op  = descriptor.get("summary_op") or {}
    status_tool = status_op.get("tool")
    if not status_tool:
        return result_text
    interval, max_interval = 3.0, 20.0
    terminal = {"done", "crashed", "unknown"}
    state = "running"
    for _ in range(100_000):  # generous safety bound; runs reach a terminal state
        await asyncio.sleep(interval)
        interval = min(interval * 1.5, max_interval)
        try:
            raw = await agent._run_tool(status_tool, dict(status_op.get("args") or {}))
            payload = json.loads(raw) if isinstance(raw, str) else (raw or {})
            state = str(payload.get("state") or "")
        except Exception:
            continue  # transient poll failure — retry next tick
        emit({"type": "status", "text": f"background run: {state}"})
        if state in terminal:
            break
    summary_text = ""
    summary_tool = summary_op.get("tool")
    if summary_tool:
        try:
            summary_text = await agent._run_tool(
                summary_tool, dict(summary_op.get("args") or {}))
        except Exception:
            summary_text = ""
    note = f"\n\n[background:awaited] Run reached state '{state}'."
    if summary_text:
        note += f" Results:\n{summary_text}"
    return result_text + note

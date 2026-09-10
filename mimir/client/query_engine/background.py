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
import logging
from typing import Any

from ..event_sink import emit
from ..context.capabilities import BACKGROUNDABLE, has_cap
from ..tool_execution.formatter import parse_tool_payload

logger = logging.getLogger(__name__)


def _maybe_emit_open_editor(result_text: str) -> None:
    """Emit an ``open_editor`` UI event when a tool result asks for a file to be
    opened in the editor.

    Shape-driven (like the exec preview): a result payload carrying
    ``open_in_editor: true`` plus an absolute ``path`` opts in — the loop does not
    hard-code any tool name. Used so a written plan (.md) pops open in VS Code for
    the user to read. Best-effort: any parse failure is silently ignored.

    Read through ``parse_tool_payload``, like the exec preview, because a result is an
    envelope followed by its text blocks and then whatever the executor appended. A
    bare ``json.loads`` sees all of that as one broken document and returns nothing —
    so the editor would stop opening the moment a post-tool annotation appeared.
    """
    payload = parse_tool_payload(result_text) if isinstance(result_text, str) else None
    if not isinstance(payload, dict) or not payload.get("open_in_editor"):
        return
    path = payload.get("path")
    if isinstance(path, str) and path.strip():
        emit({"type": "open_editor", "path": path})


def _detect_background_job(name: str, result_text: str, agent: Any) -> dict | None:
    """Return a ``background_job`` descriptor from a BACKGROUNDABLE tool result, else None.

    Shape-driven like ``_maybe_emit_open_editor`` — no tool name is hard-coded, and
    the payload is read through ``parse_tool_payload`` for the same reason: a launched
    run must not stop being watched because something was appended after its JSON.
    """
    if not has_cap(name, BACKGROUNDABLE, agent.tool_caps):
        return None
    payload = parse_tool_payload(result_text) if isinstance(result_text, str) else None
    if not isinstance(payload, dict):
        return None
    descriptor = payload.get("background_job")
    return descriptor if isinstance(descriptor, dict) else None


def _maybe_register_background_job(
    name: str, result_text: str, agent: Any, descriptor: dict | None = None,
) -> tuple[str, bool]:
    """Register a completion watcher for a detached run and tell the model to yield.

    Registration is delegated to the optional front-end hook
    ``agent._register_background_job`` (the WebSocket worker sets it; CLI does not),
    mirroring the ``_poll_steer``/``_cancel_flag`` optional-hook pattern.

    Returns ``(result, registered)``. The result carries the "you were backgrounded,
    end your turn" note **only when a watcher is actually holding the run**, and
    ``registered`` says so, because that is the one thing the caller cannot infer from
    the text: it decides whether the turn may end on the promise of a resume or must
    fall back to ``_await_background_job``. Returning only the text is what let a
    declined registration read exactly like a successful one — see the caller.

    *descriptor* lets a caller that already detected one pass it in rather than have
    the result parsed a second time; omitted, it is detected here as before.

    Best-effort: every failure path returns the result unchanged, and says so in the
    log. A watcher that was never posted used to leave no trace at all, which is why a
    run could finish into silence with nothing anywhere to explain it.
    """
    if descriptor is None:
        descriptor = _detect_background_job(name, result_text, agent)
    if descriptor is None:
        return result_text, False
    job_key = descriptor.get("job_key", "?")
    hook = getattr(agent, "_register_background_job", None)
    if not hook:
        # CLI: there is no watcher to register with. Not a failure — the caller waits
        # the run out in-turn instead.
        logger.debug("no background-job watcher hook; job %r will be awaited in-turn",
                     job_key)
        return result_text, False
    try:
        registered = bool(hook(descriptor))
    except Exception:
        logger.warning("background watcher registration raised for job %r — "
                       "falling back to an in-turn await", job_key, exc_info=True)
        return result_text, False
    if not registered:
        logger.warning("background watcher registration declined job %r — "
                       "falling back to an in-turn await", job_key)
        return result_text, False
    # Order matters: waiting is the *last* resort, not the first instruction. Read
    # the other way round, a model with plenty left to do would stop dead on a
    # two-hour build rather than get on with the rest of the task.
    note = (
        f"\n\n[background] This run is now tracked as background job '{job_key}'. "
        "A watcher resumes you automatically with its results when it completes, so "
        "do NOT poll its status and do NOT sleep waiting for it. If there is other "
        "useful work in this task, carry on with it now. If the only thing left is "
        "waiting for this run, end your turn and say what you are waiting for."
    )
    return result_text + note, True


async def _await_background_job(descriptor: dict, agent: Any, result_text: str) -> str:
    """Efficiently wait out a detached run in-turn, then append its results.

    The CLI has no persistent worker loop to host a watcher, so instead of the agent
    burning a model call per status poll, we poll the descriptor's read-only status op
    here with ``asyncio.sleep`` (zero model calls) until the run is terminal, fetch the
    summary, and fold it into the tool result. The launch tool already returned, so
    this runs outside the per-tool timeout. Best-effort: failures return what we have.

    The polls carry ``record_observations=False``: a watcher tick is the client asking
    a question on its own account, not a step the model took.
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
            raw = await agent._run_tool(status_tool, dict(status_op.get("args") or {}),
                                        record_observations=False)
            # parse_tool_payload, not json.loads: a status result is an envelope plus
            # whatever was appended to it, and a poll that cannot read its own answer
            # is a run that never reaches a terminal state.
            payload = parse_tool_payload(raw) if isinstance(raw, str) else (raw or {})
            state = str((payload or {}).get("state") or "")
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
                summary_tool, dict(summary_op.get("args") or {}),
                record_observations=False)
        except Exception:
            summary_text = ""
    note = f"\n\n[background:awaited] Run reached state '{state}'."
    if summary_text:
        note += f" Results:\n{summary_text}"
    return result_text + note

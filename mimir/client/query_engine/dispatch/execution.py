"""Running one tool call: its budget, its snapshots and diff, its row, and what its
result says about runs that outlive it."""
from __future__ import annotations

import asyncio
import difflib
import logging
import os
import time

from ...protocol import human_pause
from ...config.limits import AUTO_VALIDATION_TIMEOUT_SECS as _AUTO_VALIDATION_TIMEOUT_SECS
from ...tools.capabilities import timeout_for
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from ...protocol.event_sink import emit
from ....shared.jobs import job_key_of
from ...protocol.tool_result import json_error_payload
from ...tools.exec_preview import extract_exec_preview
from ...tool_execution.executor import run_post_tool_annotations
from ...tools.file_target import file_target
from ...tools.math_preview import extract_math_preview
from ...tools.display import error_detail, summarize_tool_result
from ..jobs.background import (
    await_background_job,
    detect_background_job,
    maybe_emit_open_editor,
    maybe_register_background_job,
    maybe_rewatch_background_jobs,
)
from ..deferral import CURRENT_CALL_ID
from .admission import Call, reads_progress_of_a_watched_run, watched_progress_note

logger = logging.getLogger(__name__)


async def _await_tool(coro, timeout: float):
    """``asyncio.wait_for``, extended by the time the thread spent in ``human_pause``."""
    task = asyncio.ensure_future(coro)
    baseline = human_pause.elapsed()
    remaining = timeout
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=remaining)
            if done:
                return task.result()
            paused = human_pause.elapsed() - baseline
            if paused <= 0:   # the budget went to the tool itself — a real timeout
                task.cancel()
                raise asyncio.TimeoutError
            baseline += paused
            remaining = paused
    except asyncio.CancelledError:
        task.cancel()   # wait() leaves the task running; wait_for would have killed it
        raise


# What anyio raises once the stdio transport of an MCP server is gone — the server's
# process died, or its exit stack was unwound while this agent still held the session.
# Matched by name rather than by import: these come up through mcp's own stack, and the
# string is what the user sees, so a rename there must not silently fall back.
_DEAD_TRANSPORT = ("ClosedResourceError", "BrokenResourceError", "EndOfStream")


def _dead_transport_hint(name: str, exc: BaseException, agent: AgentPort) -> str | None:
    """The hint for a tool whose server is gone (it never recovers), or None."""
    if type(exc).__name__ not in _DEAD_TRANSPORT:
        return None
    owner = (getattr(agent, "tool_owner", None) or {}).get(name) or "its"
    siblings = sorted(
        tool for tool, server in (getattr(agent, "tool_owner", None) or {}).items()
        if server == owner and tool != name
    )
    also = f" The same goes for {', '.join(siblings[:6])}." if siblings else ""
    return (
        f"The connection to the '{owner}' server is gone — its process died, or it was "
        f"closed while this turn was running. It is not reconnected, so retrying this "
        f"call, or any other tool of that server, fails the same way.{also} Do not retry: "
        f"stop, tell the user the '{owner}' server died and that this conversation needs "
        f"a restart to get it back, and say what was left unfinished."
    )


class _ResultRow:
    """The call's ``tool_result`` row. Time the user spends on an approval card is not
    the tool's: it is left out of the reported duration (and of the budget, see
    :func:`_await_tool`)."""

    def __init__(self, name: str, call_id: str) -> None:
        self.name, self.call_id = name, call_id
        self.started = time.perf_counter()
        self.paused_at_start = human_pause.elapsed()

    def emit(self, ok: bool, summary: str, *, exec_info: dict | None = None,
             error: str | None = None, math_info: dict | None = None,
             target: dict | None = None) -> None:
        waited = human_pause.elapsed() - self.paused_at_start
        event: dict = {
            "type": "tool_result", "id": self.call_id, "name": self.name, "ok": ok,
            "summary": summary,
            "duration_ms": max(0, int((time.perf_counter() - self.started - waited) * 1000)),
        }
        # A terminal in/out panel, an equation, the file and lines touched.
        for key, value in (("exec", exec_info), ("math", math_info), ("target", target)):
            if value is not None:
                event[key] = value
        # The full error text: the summary is a clipped one-liner.
        if not ok and error:
            event["error"] = error
        emit(event)

    def fail(self, message: str, hint: str, summary: str, error: str | None = None) -> str:
        """A failed row, and the error payload the model reads."""
        self.emit(False, summary, error=error or f"{message}\n\nHint: {hint}")
        return json_error_payload(message, hint=hint)


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _snapshot(agent: AgentPort, targets: list[str]) -> dict[str, str | None]:
    """Each target's content just before this call.

    Two baselines: the batch keeps the FIRST content seen, for the approval card's
    one diff; the per-call diff needs the content before THIS call.
    """
    before: dict[str, str | None] = {}
    for path in targets:
        agent.approvals.batch.record_snapshot(path)
        before[path] = _read_text(path)
    return before


def _emit_diffs(before_by_path: dict[str, str | None]) -> None:
    """One ``diff`` event per file the call changed."""
    for path, before in before_by_path.items():
        after = _read_text(path)
        if before is None and after is None:
            continue
        rel = os.path.relpath(path)
        # keepends lines carry their newline: default lineterm and "".join.
        patch = "".join(difflib.unified_diff(
            (before or "").splitlines(keepends=True), (after or "").splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}"))
        if not patch:
            continue
        entry: dict = {"type": "diff", "file": rel, "patch": patch}
        # New for THIS call; a later edit of a file created earlier is an edit.
        if before is None:
            entry["is_new"] = True
        emit(entry)


async def _execute(agent: AgentPort, call: Call, execution_context: ExecutionContext,
                   budget: float) -> str:
    """The call under its budget, its id stamped on the task so a card raised inside it
    names this call. Post-write validation has its own budget."""
    token = CURRENT_CALL_ID.set(call.call_id)
    try:
        return await _await_tool(
            agent.run_tool(call.name, call.args, execution_context=execution_context,
                           run_auto_validation=False, call_id=call.call_id),
            budget)
    finally:
        CURRENT_CALL_ID.reset(token)


async def _annotate(agent: AgentPort, call: Call, result: str,
                    execution_context: ExecutionContext) -> str:
    """Advisory annotations, separately budgeted: a timeout or error drops them."""
    try:
        return result + await asyncio.wait_for(
            run_post_tool_annotations(agent, call.name, call.args, result, execution_context),
            timeout=_AUTO_VALIDATION_TIMEOUT_SECS)
    except Exception:
        logger.debug("post-tool annotations failed and were skipped", exc_info=True)
        return result


async def _follow_runs(agent: AgentPort, call: Call, result: str, ok: bool) -> str:
    """What the result says about runs that outlive the call.

    A launched run is held by a watcher, else waited out in-turn. Runs reported as still
    going are re-armed when unwatched, failed calls included (an abandoned child reports
    through an error).
    """
    maybe_emit_open_editor(result)
    watched_key = (reads_progress_of_a_watched_run(agent, call.name, call.args)
                   if ok and isinstance(result, str) else None)
    if watched_key:
        result += watched_progress_note(watched_key)
    descriptor = detect_background_job(call.name, result, agent)
    if descriptor is not None:
        result, registered = maybe_register_background_job(call.name, result, agent, descriptor)
        if registered:
            # Ties the row to the run, so its progress can be matched to it.
            emit({"type": "tool_backgrounded", "id": call.call_id,
                  "job_key": job_key_of(descriptor)})
        else:
            result = await await_background_job(descriptor, agent, result)
    if isinstance(result, str):
        result = maybe_rewatch_background_jobs(call.name, result, agent)
    return result


async def run_call(agent: AgentPort, call: Call, execution_context: ExecutionContext) -> str:
    """Run one call and return its result text, reporting its row as it goes.

    Never raises for the tool's own failure — a timeout, a crashed server, a payload
    that fails to decode — which comes back as an error payload and a failed row, so one
    broken tool cannot end the turn. Cancellation still propagates.
    """
    name, args = call.name, call.args
    row = _ResultRow(name, call.call_id)
    # The wall is the tool's: a budget calibrated on a search cannot bound an agent run.
    budget = timeout_for(name, agent.tool_caps)
    try:
        targets = agent.get_tool_file_targets(name, args)
        before_by_path = _snapshot(agent, targets)
        result = await _execute(agent, call, execution_context, budget)
        ok, summary = summarize_tool_result(name, result, agent.tool_caps)
        if ok:
            result = await _annotate(agent, call, result, execution_context)
            _emit_diffs(before_by_path)
        row.emit(
            ok, summary, exec_info=extract_exec_preview(result, args),
            error=None if ok else error_detail(result),
            math_info=extract_math_preview(result) if ok else None,
            target=file_target(name, args, result, agent.tool_caps) if ok else None,
        )
        return await _follow_runs(agent, call, result, ok)

    except asyncio.TimeoutError:
        return row.fail(f"Tool '{name}' timed out after {budget}s.",
                        "The operation took too long; consider a narrower query or a "
                        "read-only alternative.",
                        f"timed out after {budget}s")

    except Exception as exc:
        # A dead server or an undecodable response is an ordinary tool failure.
        detail = f"{type(exc).__name__}: {exc}".strip().rstrip(":").strip()
        hint = _dead_transport_hint(name, exc, agent) or (
            "The tool's server process may have crashed or become unreachable. Retry "
            "once; if it recurs, that server likely needs attention (check its startup "
            "and imports).")
        return row.fail(
            f"Tool '{name}' failed to execute: {detail}", hint,
            detail.splitlines()[0][:100] if detail else "tool call failed",
            error=None if detail else "The tool call failed with no error detail.")

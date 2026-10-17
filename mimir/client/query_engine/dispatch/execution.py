"""Running one tool call: its budget, its snapshots and diff, its row, and what its
result says about runs that outlive it."""
from __future__ import annotations

import asyncio
import difflib
import logging
import os
import time

from ... import human_pause
from ...config.limits import AUTO_VALIDATION_TIMEOUT_SECS as _AUTO_VALIDATION_TIMEOUT_SECS
from ...context.capabilities import timeout_for
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from ...event_sink import emit
from ...protocol.tool_result import json_error_payload
from ...tool_execution.exec_preview import extract_exec_preview
from ...tool_execution.executor import run_post_tool_annotations
from ...tool_execution.file_target import file_target
from ...tool_execution.math_preview import extract_math_preview
from ...tool_execution.tool_status_messages import error_detail, summarize_tool_result
from ..background import (
    _await_background_job,
    _detect_background_job,
    _maybe_emit_open_editor,
    _maybe_register_background_job,
    _maybe_rewatch_background_jobs,
)
from ..deferral import CURRENT_CALL_ID
from .admission import Call, _reads_progress_of_a_watched_run, _watched_progress_note

logger = logging.getLogger(__name__)


async def _await_tool(coro, timeout: float):
    """``asyncio.wait_for``, except time spent waiting on the user doesn't count.

    The approval prompt is raised from inside the tool call, so a plain ``wait_for``
    charged the user's thinking time to the tool's budget: a command approved after
    two minutes came back "timed out after 120s" without ever having run. Here the
    budget is re-extended by however long the thread sat in ``human_pause`` — the
    prompt blocks the loop thread, so the extension is applied when it resumes.
    """
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
    """The hint for a tool whose server is gone, or None when it is a different failure.

    Worth telling apart from every other tool error because the advice is the opposite:
    a dead transport never recovers. Nothing reconnects an MCP server, so every later
    call to any tool of that server fails identically — "retry once" spends the turn on
    calls that cannot work, and the model deserves to be told to stop and say so instead.
    """
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


async def run_call(agent: AgentPort, call: Call, execution_context: ExecutionContext) -> str:
    """Run one call and return its result text, reporting its row as it goes.

    Never raises for the tool's own failure — a timeout, a crashed server, a payload
    that fails to decode — which comes back as an error payload and a failed row, so one
    broken tool cannot end the turn. Cancellation still propagates.
    """
    name, args, call_id = call.name, call.args, call.call_id
    started = time.perf_counter()
    # The wall is the tool's, not the loop's: a budget calibrated on a search
    # cannot bound a tool whose work is an agent run of its own.
    budget = timeout_for(name, agent.tool_caps)
    # Time the user spends on an approval card is not time the tool spent
    # working: it is subtracted from both the reported duration and the
    # timeout budget (see _await_tool), so a command approved after two
    # minutes is not reported as having timed out before it ever ran.
    paused_at_start = human_pause.elapsed()

    def _emit_result(
        ok: bool,
        summary: str,
        exec_info: dict | None = None,
        error: str | None = None,
        math_info: dict | None = None,
        target: dict | None = None,
    ) -> None:
        waited = human_pause.elapsed() - paused_at_start
        event = {
            "type": "tool_result",
            "id": call_id,
            "name": name,
            "ok": ok,
            "summary": summary,
            "duration_ms": max(0, int((time.perf_counter() - started - waited) * 1000)),
        }
        # Exec-shaped results (returncode + stdout/stderr) carry a clipped
        # display copy so the UI can render a terminal in/out panel.
        if exec_info is not None:
            event["exec"] = exec_info
        # A calculation typeset by the tool itself, shown as an equation.
        if math_info is not None:
            event["math"] = math_info
        # The file the call touched, and the lines, for the row's clickable name.
        if target is not None:
            event["target"] = target
        # Failures carry the FULL error text (the summary is a clipped one-liner
        # that reads as truncated in the row); the UI shows it in an expandable
        # panel under the row.
        if not ok and error:
            event["error"] = error
        emit(event)

    try:
        # ── PRE-EXECUTION SNAPSHOTS (GENERIC & SAFE) ──
        # Two baselines, deliberately: `record_snapshot` keeps the FIRST content seen
        # per path so the approval card can show the whole batch as one diff, while
        # the event emitted below is per call and needs the content as it was just
        # before THIS call. Reading the batch baseline for both is what made every
        # edit re-diff against the state at the start of the batch — for a file
        # created in-session that baseline is None forever, so all 151 diff events
        # across four recorded sessions arrived as `is_new` with the entire file as
        # additions.
        targets = agent.get_tool_file_targets(name, args)
        before_by_path: dict[str, str | None] = {}
        for path in targets:
            agent.approvals.batch.record_snapshot(path)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    before_by_path[path] = fh.read()
            except OSError:
                before_by_path[path] = None

        # ── ACTUAL TOOL EXECUTION ──
        # The timeout guards the WRITE only. Post-write auto-validation is run
        # separately below under its own budget: it happens after the file is
        # already on disk, so a slow/hung validator must never be able to trip
        # this timeout and mark a successful edit as failed.
        # Stamped on the call's own task (the future copies the context when it is
        # created), so an approval or question raised inside it names this call.
        call_token = CURRENT_CALL_ID.set(call_id)
        try:
            result = await _await_tool(
                agent.run_tool(
                    name, args,
                    execution_context=execution_context,
                    run_auto_validation=False,
                    call_id=call_id,
                ),
                budget,
            )
        finally:
            CURRENT_CALL_ID.reset(call_token)

        ok, summary = summarize_tool_result(name, result, agent.tool_caps)

        # ── POST-TOOL ANNOTATIONS (ADVISORY, SEPARATELY BUDGETED) ──
        # The built-in post-write ladder plus any registered post-tool hook. Its
        # own timeout drops the annotation rather than failing the call; an error
        # or timeout is deliberately swallowed here so it can never reach the
        # broad exception handler below (which WOULD mark the tool row failed).
        # CancelledError (query cancel) is not an Exception, so it still
        # propagates.
        if ok:
            try:
                result += await asyncio.wait_for(
                    run_post_tool_annotations(agent, name, args, result, execution_context),
                    timeout=_AUTO_VALIDATION_TIMEOUT_SECS,
                )
            except Exception:
                logger.debug("_emit_result: a best-effort step failed and was skipped", exc_info=True)

        # ── POST-EXECUTION DIFF EMIT ──
        if ok and targets:
            for path in targets:
                before = before_by_path.get(path)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fh:
                        after = fh.read()
                except OSError:
                    after = None
                if before is None and after is None:
                    continue
                before_lines = (before or "").splitlines(keepends=True)
                after_lines  = (after  or "").splitlines(keepends=True)
                rel = os.path.relpath(path)
                # `keepends=True` lines already carry their newline, so the default
                # lineterm plus `"".join` is the pairing that yields one newline per
                # line. With `lineterm=""` and a `"\n".join` every line came out
                # doubled. Same pairing as the batch-status diff in ws_worker.
                diff_lines = list(difflib.unified_diff(
                    before_lines, after_lines,
                    fromfile=f"a/{rel}", tofile=f"b/{rel}",
                ))
                if diff_lines:
                    entry: dict = {
                        "type": "diff",
                        "file": rel,
                        "patch": "".join(diff_lines),
                    }
                    # New for THIS call — the file did not exist when the call began.
                    # Not "new to the batch": a second edit of a file created moments
                    # ago is an edit, and showing it as a creation buries the change.
                    if before is None:
                        entry["is_new"] = True
                    emit(entry)

        _emit_result(
            ok, summary, extract_exec_preview(result, args),
            error=None if ok else error_detail(result),
            math_info=extract_math_preview(result) if ok else None,
            target=file_target(name, args, result, agent.tool_caps) if ok else None,
        )
        _maybe_emit_open_editor(result)
        watched_key = (_reads_progress_of_a_watched_run(agent, name, args)
                       if ok and isinstance(result, str) else None)
        if watched_key:
            result += _watched_progress_note(watched_key)
        descriptor = _detect_background_job(name, result, agent)
        if descriptor is not None:
            # Branching on the hook's *existence* left the WS front-end with no
            # fallback at all: it installs the hook unconditionally, so a
            # registration that failed produced no watcher, no in-turn await and
            # no note — and the model, told nothing, polled the job by hand until
            # the step limit. What decides is whether a watcher is actually
            # holding the run.
            result, registered = _maybe_register_background_job(
                name, result, agent, descriptor)
            if registered:
                # Tie the row to the run that outlived it. Without this a row
                # whose result is not exec-shaped — an optimization run has no
                # stdout to preview — settles as an ordinary success, and nothing
                # downstream can tell that the work it names is still going, nor
                # match the watcher's progress to it.
                emit({
                    "type":    "tool_backgrounded",
                    "id":      call_id,
                    "job_key": str(descriptor.get("job_key")
                                   or descriptor.get("run_dir") or ""),
                })
            if not registered:
                # No watcher (CLI, or a registration that declined): wait it out
                # efficiently in-turn. Costs zero model calls either way.
                result = await _await_background_job(descriptor, agent, result)
        if isinstance(result, str):
            # A result reporting runs that are still going: what a status op answers
            # about work an earlier call started, and what a sub-agent out of time
            # hands back. Re-arms whatever is no longer watched, which after an
            # agent restart is all of it, and is a no-op the rest of the time.
            #
            # Neither an `elif` nor conditional on the call having succeeded. One
            # payload may both launch a run and report others still going; and the
            # payload that most needs its live runs adopted is the one that failed,
            # because a child abandoned mid-flight reports where it got to through
            # an error. Adopting a watch trusts no state either way — a descriptor
            # must name a job and an op to poll it with (`_is_rewatchable`), and a
            # run that is not there is retired as unknown a few ticks later.
            result = _maybe_rewatch_background_jobs(name, result, agent)
        return result

    except asyncio.TimeoutError:
        _emit_result(
            False, f"timed out after {budget}s",
            error=(
                f"Tool '{name}' timed out after {budget}s.\n\n"
                "Hint: The operation took too long; consider a narrower query "
                "or a read-only alternative."
            ),
        )
        return json_error_payload(
            f"Tool '{name}' timed out after {budget}s.",
            hint=(
                "The operation took too long; consider a narrower query or a "
                "read-only alternative."
            ),
        )

    except Exception as exc:
        # Any other exception here — most often a dead server subprocess whose
        # MCP session broke (session.call_tool raises), or a response that fails
        # to decode — would otherwise bubble out of the entire query and surface
        # as a single opaque error line, killing the turn. Convert it into a
        # normal tool failure instead: the model gets actionable feedback, the UI
        # shows a proper failed tool row, and the error payload is persisted in
        # history like any other tool result (so it survives in the session log).
        # CancelledError is a BaseException and is deliberately NOT caught here,
        # so query cancellation still propagates.
        detail = f"{type(exc).__name__}: {exc}".strip().rstrip(":").strip()
        hint = _dead_transport_hint(name, exc, agent) or (
            "The tool's server process may have crashed or become "
            "unreachable. Retry once; if it recurs, that server likely "
            "needs attention (check its startup and imports)."
        )
        _emit_result(
            False,
            detail.splitlines()[0][:100] if detail else "tool call failed",
            error=(f"Tool '{name}' failed to execute: {detail}\n\nHint: {hint}")
            if detail else "The tool call failed with no error detail.",
        )
        return json_error_payload(
            f"Tool '{name}' failed to execute: {detail}",
            hint=hint,
        )

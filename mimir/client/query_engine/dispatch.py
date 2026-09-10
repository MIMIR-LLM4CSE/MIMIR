"""Tool-call dispatch for one model step, plus the spin/dedup guards.

``_dispatch_tool_calls`` runs all tool calls for a step (parallel reads, sequential
writes) with dedup + a repeated-failing-call guard; ``_post_dispatch_inject`` adds
post-dispatch correctives. The guard's thresholds and synthetic payload live here, next
to the dispatch that uses them; its corrective *copy* is in ``guardrails.workflow``.
Extracted from ``agent_loop.py``.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import os
import json
import time
from typing import Any

from ..config.constants import (
    AUTO_VALIDATION_TIMEOUT_SECS as _AUTO_VALIDATION_TIMEOUT_SECS,
    IDENTICAL_REPEAT_THRESHOLD,
    NUDGE_MAX_TODO_TICK,
)
from ..event_sink import emit
from .. import human_pause
from ..context.capabilities import EDIT, has_cap, label_for, scope_spec, timeout_for
from ..context.execution_context import loop_control, nudge_count
from ..tool_execution.normalizer import _make_hashable
from ..tool_execution.executor import run_post_tool_annotations
from ..tool_execution.exec_preview import extract_exec_preview
from ..tool_execution.tool_status_messages import (
    tool_status_message,
    tool_arg_preview,
    dedup_row_detail,
    shorten_display_args,
    summarize_tool_result,
    error_detail,
)
from ..guardrails.workflow import (
    handback_corrective_message,
    handback_required,
    handback_scopes,
    moving_test_corrective_message,
    repeat_corrective_message,
)
from ..guardrails.nudges import inject_reminder, maybe_inject_env_resolution
from .streaming import _to_dict
from .background import (
    _maybe_emit_open_editor,
    _detect_background_job,
    _maybe_register_background_job,
    _await_background_job,
)


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



# ── Guards for a run something else is already watching ───────────────────────
#
# Both read one deterministic fact — the set of watchers currently alive — through the
# optional ``agent._watched_background_jobs`` hook (the WS worker sets it; the CLI does
# not, and then both guards abstain). They live in the dispatch rather than in a policy
# gate on purpose: the dispatch is the *model's* path, while the watcher's own probes
# and the CLI's in-turn await call ``agent._run_tool`` directly. A gate would see both
# and, refusing the watcher's probe, would break the very mechanism these protect.


def _watched_jobs(agent: Any) -> list[dict]:
    """Descriptors of the runs a watcher is holding right now. Best-effort, never raises."""
    hook = getattr(agent, "_watched_background_jobs", None)
    if not hook:
        return []
    try:
        jobs = hook() or []
    except Exception:
        return []
    return [d for d in jobs if isinstance(d, dict)]


def _asks_whether_a_watched_run_is_done(
    agent: Any, name: str, args: dict,
) -> str | None:
    """The job_key this call is asking the state of, when a watcher already answers it.

    Shape-driven: the comparison is against the descriptor's own ``status_op`` and
    ``summary_op``, which is registry data the server put there — no tool name is
    spelled out here, for the same reason the watcher can poll generically.

    The summary op is checked first and always allowed. Reading a run's output while it
    goes is progress ("which target is this build on"), which the watcher does not
    report until the end; the state op is the question "is it finished yet", which the
    wake answers on its own. Only the second one is refused.

    Matching is containment, not equality: the descriptor's args are the ones that
    identify the job, and a caller that adds an explicit default (an ``op="status"``
    the descriptor left implicit) is asking the same question. Equality would have let
    exactly that spelling through.
    """
    def _matches(op: Any) -> bool:
        if not isinstance(op, dict) or op.get("tool") != name:
            return False
        op_args = op.get("args")
        if not isinstance(op_args, dict):
            return False
        return all(args.get(k) == v for k, v in op_args.items())

    for descriptor in _watched_jobs(agent):
        if _matches(descriptor.get("summary_op")):
            return None            # progress, not "are we there yet" — always allowed
        if _matches(descriptor.get("status_op")):
            return str(descriptor.get("job_key") or "?")
    return None


def _waits_by_blocking_the_turn(agent: Any, name: str, args: dict) -> bool:
    """True when this call's command opens by doing nothing but passing time.

    Registry-driven on both halves: the tool that carries a raw command line is the one
    declaring a ``command_prefix`` scope (the test ``gates._shell_command_args`` and
    ``observations._carries_shell_command`` already make), and the program of the
    leading segment comes from the shared segmenter, which skips ``VAR=val`` assignments
    and wrappers of its own. ``sleep`` is named as a POSIX program, the way this
    codebase already names interpreters and wrappers — never as a tool.

    Only the *leading* segment counts. ``./run.sh --sleep 60`` runs a script, and
    ``make; sleep 1`` has already done the work; neither is a turn spent waiting.
    Fail-open on a command the shared parser refuses, as every other shell guard does.
    """
    spec = scope_spec(name, getattr(agent, "tool_caps", None))
    if not spec or spec.get("kind") != "command_prefix":
        return False
    from ..guardrails.policy.bash_classify import shell_segments
    from ..guardrails.policy.gates import _segment_program
    for arg in (spec.get("args") or ("command",)):
        command = args.get(arg)
        if not isinstance(command, str) or not command.strip():
            continue
        segments = shell_segments(command, allow_expansion=True)
        if not segments:
            continue
        program = _segment_program(list(segments[0].argv))
        if program and os.path.basename(program) == "sleep":
            return True
    return False


def _watched_poll_blocked_payload(job_key: str) -> str:
    """Synthetic result for a call asking whether a watched run has finished."""
    return json.dumps({
        "status": "error",
        "error": (
            f"Background job '{job_key}' is being watched, so its state was not read. "
            "You will be resumed automatically with its results the moment it finishes."
        ),
        "hint": (
            "Do not ask again whether it is done. If there is other useful work in this "
            "task, carry on with it now. If the only thing left is waiting for this run, "
            "end your turn and say what you are waiting for. Reading the run's output "
            "for progress is still available if you need to see how far along it is."
        ),
    })


def _blocking_wait_payload() -> str:
    """Synthetic result for a turn spent waiting on a run something else is watching."""
    return json.dumps({
        "status": "error",
        "error": (
            "This command opens by waiting, and a background job is already being "
            "watched for you, so it was not run. Blocking the turn this way also delays "
            "the completion notice it is waiting for, and any message the user sends "
            "meanwhile."
        ),
        "hint": (
            "You will be resumed automatically when the run finishes. If there is other "
            "useful work in this task, carry on with it now. If the only thing left is "
            "waiting, end your turn and say what you are waiting for."
        ),
    })


async def _dispatch_tool_calls(
    tool_calls: list,
    agent: Any,
    messages: list[dict],
    execution_context: dict,
) -> None:
    """Execute all tool calls for one model step, in parallel where possible.

    Independent tool calls issued within the same model response are dispatched
    concurrently via asyncio.gather.  Results are appended to messages in the
    original order so the conversation history stays deterministic.
    """
    normalized: list[tuple[str, dict, str]] = []
    seen_calls: set[tuple] = set()
    # Per-query loop-control state (dedup + spin guards), kept in a dedicated object
    # outside the ExecutionContext schema. Created lazily on first dispatch.
    lc = loop_control(execution_context)
    # Persistent cross-step dedup set so the same write call cannot be re-executed in
    # a later step (e.g. after a nudge or a step-counter reset).
    cross_step_write_calls: set[tuple] = lc.write_calls
    # Per-key count of identical FAILED non-write dispatches this query, used to
    # backstop the repeated-failing-call spin (see _repeat_blocked_payload).
    call_fails: dict = lc.call_fails
    # call_id -> synthetic result, for calls hard-blocked because they have already
    # failed identically too many times. Kept in `normalized` so a tool message is
    # still emitted for the model's tool_call id, but never actually executed.
    blocked_results: dict[str, str] = {}
    for idx, tc in enumerate(tool_calls):
        tc = _to_dict(tc)
        fn = _to_dict(tc.get("function", {}))
        name = fn.get("name", "")
        args = agent._normalize_arguments(fn.get("arguments") or {})
        call_id = tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else f"call_{idx}"
        # Deduplicate: skip exact (name, args) duplicates within one step.
        key = (name, _make_hashable(args))
        if key in seen_calls:
            continue
        seen_calls.add(key)
        is_write = agent._is_write_tool(name) or has_cap(name, EDIT, agent.tool_caps)
        # Cross-step dedup: skip write tools whose exact call was already
        # dispatched in a previous step of this query.
        if is_write:
            if key in cross_step_write_calls:
                continue
            cross_step_write_calls.add(key)
        # Hard backstop for non-write tools: an identical call that has already failed
        # HARD_REPEAT_LIMIT times is not executed again — return a synthetic error so
        # the model gets feedback instead of silently spinning to the step ceiling.
        elif call_fails.get(key, 0) >= HARD_REPEAT_LIMIT:
            blocked_results[call_id] = _repeat_blocked_payload(name, call_fails[key])
            emit({"type": "status", "text": f"  ⛔ Blocking repeated failing call: {name}"})
        # A run a watcher is already holding answers both of these on its own: asking
        # whether it is done, and spending the turn waiting for it. Refused rather than
        # nudged, because the fact is a lookup and not a judgement — and because the
        # standing instruction is delivered once, at launch, and long buried by the time
        # it is disregarded.
        #
        # Deliberately outside the chain above rather than another `elif`: that chain
        # branches on whether the tool writes, which has nothing to do with whether a
        # run is being watched. A backgroundable tool that also edited a file would
        # silently escape a guard hung off its `else`. Never overrides a block already
        # decided — the repeat guard's answer is the more specific one.
        if call_id not in blocked_results:
            watched_key = _asks_whether_a_watched_run_is_done(agent, name, args)
            if watched_key:
                blocked_results[call_id] = _watched_poll_blocked_payload(watched_key)
                emit({"type": "status", "text":
                      "  ⛔ Already watching that run — you will be resumed when it ends"})
            elif _waits_by_blocking_the_turn(agent, name, args):
                blocked_results[call_id] = _blocking_wait_payload()
                emit({"type": "status", "text":
                      "  ⛔ Not waiting in-turn — a watcher already holds that run"})
        normalized.append((name, args, call_id))

    for name, args, call_id in normalized:
        display_name, display_args = agent._rewrite_tool_for_context(name, args)
        # Label precedence: the capability-declared template (server-side
        # `label="Reading file: {path}"`) wins; the client status map is the
        # fallback for tools that declare none.
        # Paths are shortened to their file name for the row: tools carry absolute
        # paths now, and a row reading "Reading file: /shared/.../mimir/client/foo.py"
        # buries the only token the user is scanning for. Approval prompts keep the
        # absolute path — see tool_status_messages._relpath.
        row_args = shorten_display_args(display_name, display_args, agent.tool_caps)
        row_label = (label_for(display_name, row_args, agent.tool_caps)
                     or tool_status_message(display_name, row_args))
        # Drop a detail that just repeats the label (e.g. the basename when the
        # label already shows the full path) — see dedup_row_detail.
        row_detail = dedup_row_detail(
            row_label, tool_arg_preview(display_name, row_args))
        emit({
            "type": "tool_call",
            "id": call_id,
            "name": display_name,
            "label": row_label,
            "detail": row_detail,
        })

    
    async def _run_with_timeout(name: str, args: dict, call_id: str) -> str:
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
                agent.approvals.record_snapshot(path)
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
            result = await _await_tool(
                agent._run_tool(
                    name, args,
                    execution_context=execution_context,
                    run_auto_validation=False,
                    call_id=call_id,
                ),
                budget,
            )

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
                    pass

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
            )
            _maybe_emit_open_editor(result)
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
                if not registered:
                    # No watcher (CLI, or a registration that declined): wait it out
                    # efficiently in-turn. Costs zero model calls either way.
                    result = await _await_background_job(descriptor, agent, result)
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
            return (
                f'{{"status": "error", "error": "Tool \'{name}\' timed out after '
                f'{budget}s.", "hint": "The operation took too long; '
                f'consider a narrower query or a read-only alternative."}}'
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
            _emit_result(
                False,
                detail.splitlines()[0][:100] if detail else "tool call failed",
                error=(
                    f"Tool '{name}' failed to execute: {detail}\n\n"
                    "Hint: The tool's server process may have crashed or become "
                    "unreachable. Retry once; if it recurs, that server likely "
                    "needs attention (check its startup and imports)."
                ) if detail else "The tool call failed with no error detail.",
            )
            return json.dumps({
                "status": "error",
                "error": f"Tool '{name}' failed to execute: {detail}",
                "hint": "The tool's server process may have crashed or become "
                        "unreachable. Retry once; if it recurs, that server likely "
                        "needs attention (check its startup and imports).",
            })


    # Reads can run concurrently safely. Write tools must NOT: each does a
    # snapshot → execute → diff sequence ( _run_with_timeout above), so two
    # writes to the same file — or a write racing a read of that file — would
    # interleave nondeterministically and corrupt the captured diff (or the file
    # itself). Run all reads in parallel, then writes strictly sequentially.
    # Results are reassembled in the original `normalized` order so the appended
    # tool messages stay deterministic regardless of execution order.
    def _is_write(tool_name: str) -> bool:
        return agent._is_write_tool(tool_name) or has_cap(tool_name, EDIT, agent.tool_caps)

    results: list[str] = [""] * len(normalized)

    # Hard-blocked repeated calls never execute; their synthetic result is filled in.
    for i, (name, args, call_id) in enumerate(normalized):
        if call_id in blocked_results:
            results[i] = blocked_results[call_id]

    read_positions = [
        i for i, (name, _, call_id) in enumerate(normalized)
        if not _is_write(name) and call_id not in blocked_results
    ]
    if read_positions:
        read_results = await asyncio.gather(
            *[_run_with_timeout(*normalized[i]) for i in read_positions]
        )
        for i, result in zip(read_positions, read_results):
            results[i] = result

    for i, (name, args, call_id) in enumerate(normalized):
        if _is_write(name) and call_id not in blocked_results:
            results[i] = await _run_with_timeout(name, args, call_id)

    # Count identical non-write failures across steps and, on the first time a call
    # crosses SOFT_REPEAT_THRESHOLD, stage a one-time mid-loop corrective (consumed by
    # _post_dispatch_inject). Skips writes (own dedup) and already-blocked calls.
    warned: set = lc.repeat_warned
    for i, (result, (name, args, call_id)) in enumerate(zip(results, normalized)):
        if _is_write(name) or call_id in blocked_results:
            continue
        key = (name, _make_hashable(args))
        ok, _summary = summarize_tool_result(name, result, agent.tool_caps)
        if ok:
            # The same call returning the same answer for the third time is spin, and
            # nothing later in the turn will notice it: the failing-call guard above
            # counts only failures, and a nudge fires only once the model stops calling
            # tools — which a spinning model never does. Said as an annotation, never a
            # block: two guards that withheld or rewrote a repeated success were built
            # here before and removed, because refusing the content only sent the model
            # to read the same thing another way. Nothing is withheld here.
            digest = hashlib.sha1(str(result).encode("utf-8", "replace")).hexdigest()
            seen, count = lc.call_results.get(key, ("", 0))
            count = count + 1 if seen == digest else 1
            lc.call_results[key] = (digest, count)
            if count >= IDENTICAL_REPEAT_THRESHOLD and key not in lc.repeat_noted:
                lc.repeat_noted.add(key)
                results[i] = str(result) + (
                    f"\n\nIDENTICAL_REPEAT: this call has returned exactly this result "
                    f"{count} times in this task. It will not return anything else — "
                    f"decide with what you already have, or ask a different question."
                )
            continue
        call_fails[key] = call_fails.get(key, 0) + 1
        if call_fails[key] >= SOFT_REPEAT_THRESHOLD and key not in warned:
            warned.add(key)
            execution_context["_repeat_alert"] = (name, call_fails[key])

    # Record which files each tool message concerns, keyed by tool_call_id, so
    # _trim_tool_history can match messages to files structurally instead of by
    # fragile substring scanning. An empty list is meaningful: it marks a message
    # (e.g. a grep/bash result) as touching no tracked file, so eviction won't
    # falsely invalidate a read just because a path appears in the output text.
    tool_msg_files: dict = execution_context.setdefault("tool_msg_files", {})
    for result, (name, args, call_id) in zip(results, normalized):
        try:
            tool_msg_files[call_id] = agent.get_tool_file_targets(name, args)
        except Exception:
            tool_msg_files[call_id] = []
        messages.append({"role": "tool", "tool_call_id": call_id, "content": result})


async def _post_dispatch_inject(
    agent: Any,
    messages: list[dict],
    execution_context: dict,
    *,
    active_mode: str = "agent",
) -> None:
    """After every tool dispatch step, inject post-dispatch reminders.

    Five independent reminders: (1) mark a completed todo step done after a successful
    edit, (2) a one-time corrective when a non-write call keeps failing identically
    (staged as ``_repeat_alert`` during dispatch), (3) a one-time corrective when a test
    file's failing set keeps changing between runs (staged as ``_moving_test_alert`` by
    the observation layer), (4) a one-time stop when refusals have run the denial ladder
    to its end, and (5) the environment-resolution cascade when a call just failed on a
    missing module. This is the mid-tool-loop channel the regular nudges can't reach,
    since they only fire when the model stops calling tools — and a model that has been
    told to hand back, that is retrying against the wrong interpreter, or that is
    chasing a moving test, is by definition still calling tools.
    """
    # remind agent to mark completed step done in todo list
    success_path = execution_context.get("last_edit_success_path", "")
    if (
        success_path
        and execution_context.get("todo_written")
        and execution_context.get("todo_file_path")
    ):
        execution_context["last_edit_success_path"] = ""  # consume
        counts = execution_context.setdefault("nudge_counts", {})
        if (
            # The per-category cap every nudge in the table honours, applied here too:
            # this reminder reached the model on EVERY successful write, uncapped, which
            # in a debugging stretch is most of the turn.
            nudge_count(execution_context, "todo_tick") < NUDGE_MAX_TODO_TICK
            # And never twice running about the same file: re-editing what we just spoke
            # about is the middle of one change, not the end of a step.
            and success_path != execution_context.get("_last_todo_tick_path")
        ):
            counts["todo_tick"] = counts.get("todo_tick", 0) + 1
            execution_context["_last_todo_tick_path"] = success_path
            inject_reminder(
                messages,
                f"You just wrote {os.path.basename(success_path)} successfully. "
                "If a step in your task checklist is now fully complete, mark it done. "
                "Do NOT mark it done if more work for that step remains.",
                category="todo_tick",
                execution_context=execution_context,
            )

    # one-time corrective for a repeated identical failing call
    alert = execution_context.pop("_repeat_alert", None)
    if alert:
        name, fails = alert
        inject_reminder(
            messages, repeat_corrective_message(name, fails), category="repeat_call",
            execution_context=execution_context,
        )

    # one-time corrective for a test file whose failing set keeps changing
    moving_test = execution_context.pop("_moving_test_alert", None)
    if moving_test:
        inject_reminder(
            messages, moving_test_corrective_message(moving_test),
            category="moving_test", execution_context=execution_context,
        )

    # one-time stop once refusals reached the end of the denial ladder
    if handback_required(execution_context) and not execution_context.get("_handback_told"):
        execution_context["_handback_told"] = True
        inject_reminder(
            messages,
            handback_corrective_message(handback_scopes(execution_context)),
            category="handback",
            execution_context=execution_context,
        )

    # the env cascade, at the failure rather than a step ceiling later
    maybe_inject_env_resolution(
        agent=agent,
        active_mode=active_mode,
        execution_context=execution_context,
        messages=messages,
    )


# Repeated-failing-call guard. Nudges only fire when the model stops calling tools,
# and cross-step dedup only blocks *writes* — so a non-write call that fails can be
# re-issued identically every step until the step ceiling. These two thresholds turn
# that silent spin into (a) a one-time mid-loop corrective and (b) a hard backstop.
# (The corrective *copy* lives in guardrails.workflow; only the gating/thresholds and the
# synthetic tool-result payloads stay here, next to the dispatch that uses them.)
SOFT_REPEAT_THRESHOLD = 2   # after this many identical FAILED dispatches, inject the corrective once
HARD_REPEAT_LIMIT = 3       # once this many identical failures are recorded, block further attempts



def _repeat_blocked_payload(tool_name: str, fails: int) -> str:
    """Synthetic tool result returned in place of an over-repeated failing call."""
    return json.dumps({
        "status": "error",
        "error": (
            f"This exact call failed {fails} times already and was not retried. "
            "Repeating an identical failing call is blocked."
        ),
        "hint": (
            "Do not repeat this call. Either change the approach (different arguments, "
            "a different tool, or resolve the underlying environment/precondition), or "
            "stop and conclude clearly that you cannot proceed and why."
        ),
    })


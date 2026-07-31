"""Tool-call dispatch for one model step, plus the spin/dedup guards.

``_dispatch_tool_calls`` runs all tool calls for a step (parallel reads, sequential
writes) with dedup + repeated/redundant-call guards; ``_post_dispatch_inject`` adds
post-dispatch correctives. The failing-call and redundant-success guards (thresholds
+ synthetic payloads + history stripping) live here, next to the dispatch that uses
them; their corrective *copy* is in ``guardrails.workflow``. Extracted from
``agent_loop.py``.
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
    TOOL_CALL_TIMEOUT_SECS as _TOOL_TIMEOUT_SECS,
    AUTO_VALIDATION_TIMEOUT_SECS as _AUTO_VALIDATION_TIMEOUT_SECS,
)
from ..event_sink import emit
from .. import human_pause
from ..context.capabilities import EDIT, has_cap, label_for
from ..context.execution_context import loop_control
from ..tool_execution.normalizer import _make_hashable
from ..tool_execution.executor import run_post_write_validation
from ..tool_execution.exec_preview import extract_exec_preview
from ..tool_execution.tool_status_messages import (
    tool_status_message,
    tool_arg_preview,
    dedup_row_detail,
    shorten_display_args,
    summarize_tool_result,
    error_detail,
)
from ..guardrails.workflow import repeat_corrective_message, redundant_corrective_message
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
    # Per-key (result_hash, redundant_repeat_count) for non-write dispatches that
    # SUCCEEDED with identical result content across steps. Backstops the
    # redundant-successful-read spin (e.g. re-reading the same file unchanged over and
    # over). A changed result resets the count, so edit-then-reread stays allowed.
    call_results: dict = lc.call_results
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
        elif (call_fails.get(key, 0) >= HARD_REPEAT_LIMIT
              or getattr(agent, "_persistent_call_fails", {}).get(key, 0) >= HARD_REPEAT_LIMIT):
            n_fails = max(call_fails.get(key, 0),
                          getattr(agent, "_persistent_call_fails", {}).get(key, 0))
            blocked_results[call_id] = _repeat_blocked_payload(name, n_fails)
            emit({"type": "status", "text": f"  ⛔ Blocking repeated failing call: {name}"})
        # Hard backstop for non-write tools that keep SUCCEEDING with identical content:
        # once the same call has returned the same result REDUNDANT_HARD_LIMIT times it
        # adds no new information, so block it AND strip the intermediate repeats from
        # history (keeping only the first copy) to break the redundant-read spin.
        elif call_results.get(key, (None, 0))[1] >= REDUNDANT_HARD_LIMIT:
            blocked_results[call_id] = _redundant_blocked_payload(name, call_results[key][1])
            emit({"type": "status", "text": f"  ⛔ Blocking redundant repeated call: {name}"})
            _strip_redundant_history(messages, execution_context, key)
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
            targets = agent.get_tool_file_targets(name, args)
            for path in targets:
                agent.approvals.record_snapshot(path)

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
                ),
                _TOOL_TIMEOUT_SECS,
            )

            ok, summary = summarize_tool_result(name, result, agent.tool_caps)

            # ── POST-WRITE VALIDATION (ADVISORY, SEPARATELY BUDGETED) ──
            # Runs only for a successful edit. Its own timeout drops the advisory
            # annotation rather than failing the write; a validation error/timeout
            # is deliberately swallowed here so it can never reach the broad
            # exception handler below (which WOULD mark the tool row failed).
            # CancelledError (query cancel) is not an Exception, so it still
            # propagates.
            if ok and has_cap(name, EDIT, agent.tool_caps):
                try:
                    result += await asyncio.wait_for(
                        run_post_write_validation(agent, name, args, execution_context),
                        timeout=_AUTO_VALIDATION_TIMEOUT_SECS,
                    )
                except Exception:
                    pass

            # ── POST-EXECUTION DIFF EMIT ──
            if ok and targets:
                for path in targets:
                    before = agent.approvals._file_snapshots.get(path)
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
                    diff_lines = list(difflib.unified_diff(
                        before_lines, after_lines,
                        fromfile=f"a/{rel}", tofile=f"b/{rel}",
                        lineterm="",
                    ))
                    if diff_lines:
                        entry: dict = {
                            "type": "diff",
                            "file": rel,
                            "patch": "\n".join(diff_lines),
                        }
                        if not before_lines:  # new file — before was empty
                            entry["is_new"] = True
                        emit(entry)

            _emit_result(
                ok, summary, extract_exec_preview(result, args),
                error=None if ok else error_detail(result),
            )
            _maybe_emit_open_editor(result)
            descriptor = _detect_background_job(name, result, agent)
            if descriptor is not None:
                if getattr(agent, "_register_background_job", None):
                    result = _maybe_register_background_job(name, result, agent)
                else:
                    # CLI (no watcher): wait it out efficiently in-turn.
                    result = await _await_background_job(descriptor, agent, result)
            return result

        except asyncio.TimeoutError:
            _emit_result(
                False, f"timed out after {_TOOL_TIMEOUT_SECS}s",
                error=(
                    f"Tool '{name}' timed out after {_TOOL_TIMEOUT_SECS}s.\n\n"
                    "Hint: The operation took too long; consider a narrower query "
                    "or a read-only alternative."
                ),
            )
            return (
                f'{{"status": "error", "error": "Tool \'{name}\' timed out after '
                f'{_TOOL_TIMEOUT_SECS}s.", "hint": "The operation took too long; '
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
    # Successful non-write calls that keep returning identical content are tracked the
    # same way (redundant-read spin) via call_results.
    warned: set = lc.repeat_warned
    redundant_warned: set = lc.redundant_warned
    for result, (name, args, call_id) in zip(results, normalized):
        if _is_write(name) or call_id in blocked_results:
            continue
        ok, _summary = summarize_tool_result(name, result, agent.tool_caps)
        key = (name, _make_hashable(args))
        if not ok:
            call_fails[key] = call_fails.get(key, 0) + 1
            pcf = getattr(agent, "_persistent_call_fails", None)
            if pcf is not None:
                pcf[key] = pcf.get(key, 0) + 1
            if call_fails[key] >= SOFT_REPEAT_THRESHOLD and key not in warned:
                warned.add(key)
                execution_context["_repeat_alert"] = (name, call_fails[key])
            continue
        # Successful non-write call: clear any persistent failure count for this exact
        # call (a call that now succeeds must not stay blocked across queries), then
        # track repeats that return identical content. A changed result resets the
        # counter so edit-then-reread is never penalised.
        getattr(agent, "_persistent_call_fails", {}).pop(key, None)
        result_hash = hashlib.sha1((result or "").encode("utf-8", "replace")).hexdigest()
        prev_hash, prev_count = call_results.get(key, (None, 0))
        identical = prev_hash == result_hash
        new_count = prev_count + 1 if identical else 0
        call_results[key] = (result_hash, new_count)
        # Track the call_ids of the identical-content occurrences so the hard block can
        # strip everything after the first copy. A changed result starts a fresh chain.
        redundant_ids: dict = lc.redundant_call_ids
        if identical:
            redundant_ids.setdefault(key, []).append(call_id)
        else:
            redundant_ids[key] = [call_id]
        if new_count >= REDUNDANT_SOFT_THRESHOLD and key not in redundant_warned:
            redundant_warned.add(key)
            execution_context["_redundant_alert"] = (name, new_count)

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
) -> None:
    """After every tool dispatch step, inject post-dispatch reminders.

    Two independent reminders: (1) mark a completed todo step done after a successful
    edit, and (2) a one-time corrective when a non-write call keeps failing identically
    (staged as ``_repeat_alert`` during dispatch) — this is the mid-tool-loop channel
    the regular nudges can't reach, since they only fire when the model stops calling tools.
    """
    # remind agent to mark completed step done in todo list
    success_path = execution_context.get("last_edit_success_path", "")
    if (
        success_path
        and execution_context.get("todo_written")
        and execution_context.get("todo_file_path")
    ):
        execution_context["last_edit_success_path"] = ""  # consume
        messages.append({
            "role": "user",
            "content": (
                f"You just wrote {os.path.basename(success_path)} successfully. "
                "If a step in your task checklist is now fully complete, mark it done. "
                "Do NOT mark it done if more work for that step remains."
            ),
        })

    # one-time corrective for a repeated identical failing call
    alert = execution_context.pop("_repeat_alert", None)
    if alert:
        name, fails = alert
        messages.append({
            "role": "user",
            "content": repeat_corrective_message(name, fails),
        })

    # one-time corrective for a repeated identical SUCCESSFUL call (redundant read spin)
    redundant_alert = execution_context.pop("_redundant_alert", None)
    if redundant_alert:
        name, repeats = redundant_alert
        messages.append({
            "role": "user",
            "content": redundant_corrective_message(name, repeats),
        })


# Repeated-failing-call guard. Nudges only fire when the model stops calling tools,
# and cross-step dedup only blocks *writes* — so a non-write call that fails can be
# re-issued identically every step until the step ceiling. These two thresholds turn
# that silent spin into (a) a one-time mid-loop corrective and (b) a hard backstop.
# (The corrective *copy* lives in guardrails.workflow; only the gating/thresholds and the
# synthetic tool-result payloads stay here, next to the dispatch that uses them.)
SOFT_REPEAT_THRESHOLD = 2   # after this many identical FAILED dispatches, inject the corrective once
HARD_REPEAT_LIMIT = 3       # once this many identical failures are recorded, block further attempts

# Redundant-successful-call guard (a non-write call that keeps returning identical
# content). Tuned more aggressively than the failed-call guard: a redundant read is
# never useful, so we correct on the very first repeat and block on the second. On the
# hard block the intermediate repeats are also stripped from history (see
# _strip_redundant_history) so only the first copy of the result survives.
REDUNDANT_SOFT_THRESHOLD = 1  # repeats (beyond the first call) before the corrective is injected once
REDUNDANT_HARD_LIMIT = 2      # repeats recorded before the next identical call is blocked + history stripped


def _tc_id(tc: Any) -> Any:
    """Extract the id from a tool_call entry (dict or object form)."""
    if isinstance(tc, dict):
        return tc.get("id")
    return getattr(tc, "id", None)


def _strip_redundant_history(messages: list[dict], execution_context: dict, key: tuple) -> None:
    """Remove redundant repeated tool exchanges from history, keeping only the first.

    When a non-write call is hard-blocked for returning identical content repeatedly,
    every occurrence after the first adds nothing but reinforces the loop (and bloats
    the window). This drops those intermediate exchanges — both the assistant
    ``tool_calls`` entry and its paired ``tool`` result — so the conversation keeps a
    single copy of the result. Assistant/tool pairing is preserved: a tool_call is only
    removed together with its result, and an assistant turn that existed solely to issue
    the stripped call(s) is dropped entirely.
    """
    lc = loop_control(execution_context)
    call_ids = lc.redundant_call_ids.get(key, [])
    if len(call_ids) <= 1:
        return  # nothing to collapse — only the keeper exists
    strip_ids = set(call_ids[1:]) - {call_ids[0]}  # keep call_ids[0]; never strip the keeper
    if not strip_ids:
        return

    kept: list[dict] = []
    for m in messages:
        role = m.get("role")
        # Drop the redundant tool-result messages outright.
        if role == "tool" and m.get("tool_call_id") in strip_ids:
            continue
        # Remove stripped tool_calls from assistant turns; drop the turn if it becomes
        # empty (no surviving tool_calls and no textual content).
        if role == "assistant" and isinstance(m.get("tool_calls"), list) and m.get("tool_calls"):
            remaining = [tc for tc in m["tool_calls"] if _tc_id(tc) not in strip_ids]
            if len(remaining) != len(m["tool_calls"]):
                if not remaining and not str(m.get("content") or "").strip():
                    continue  # assistant turn only issued the stripped call(s)
                m = {k: v for k, v in m.items() if k != "tool_calls"}
                if remaining:
                    m["tool_calls"] = remaining
        kept.append(m)

    messages[:] = kept

    # Drop stripped ids from the file-tracking map and collapse tracking to the keeper.
    tool_msg_files = execution_context.get("tool_msg_files", {})
    for cid in strip_ids:
        tool_msg_files.pop(cid, None)
    lc.redundant_call_ids[key] = call_ids[:1]


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


def _redundant_blocked_payload(tool_name: str, repeats: int) -> str:
    """Synthetic tool result returned in place of an over-repeated redundant call.

    Fires for a non-write call that has already SUCCEEDED with identical result content
    repeatedly — re-running it cannot surface anything new. Uses a neutral ("skipped")
    status rather than "error" so the model reads it as guidance to proceed, not as a
    failure to apologize for or retry.
    """
    return json.dumps({
        "status": "skipped",
        "reason": (
            f"This exact call already returned the same result {repeats} times, so it was "
            "not run again — repeating it adds no new information. This is not a failure."
        ),
        "next": (
            "You already have this result in the conversation above — re-read it there. "
            "Move on: act on what you found, read a different file or line range, or "
            "conclude. Do not issue the same call again."
        ),
    })

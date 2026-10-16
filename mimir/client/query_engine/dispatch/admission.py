"""Which of a step's calls run: the guards a call meets before anything executes.

Every call the model made is answered — run, or refused with a payload saying why —
so the history keeps one result per call. Refused here: an exact duplicate within the
step, a write already applied earlier in the query, a call whose arguments were cut
off, a call that has already failed identically too often, and the two ways of waiting
on a run a watcher already holds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from ...config.limits import HARD_REPEAT_LIMIT
from ...context.capabilities import EDIT, has_cap, shell_command_args
from ...event_sink import emit
from ...context.bash_classify import segment_program, shell_segments
from ...protocol.tool_call import DOING_ARG, UNPARSED_ARGUMENTS, as_dict
from ...protocol.tool_result import json_error_payload
from ...tool_execution.normalizer import _make_hashable
from ...tool_execution.tool_status_messages import clip_doing
from ...context.execution_context import ExecutionContext


@dataclass
class Call:
    """One call the model made, as the dispatcher handles it."""

    name: str
    args: dict
    call_id: str
    # The model's own sentence about the call (DOING_ARG), taken out of the arguments:
    # it is display, and everything downstream — dedup, policy, execution — must see
    # the arguments the tool will actually be given.
    doing: str = ""
    # A synthetic result answering the call instead of running it.
    refused: str | None = None

    @property
    def key(self) -> tuple:
        return (self.name, _make_hashable(self.args))


@dataclass
class Admission:
    """A step's calls after the guards: the ones answered here or run, in order, and the
    ones not run at all, each with the reason its tool message will give."""

    calls: list[Call] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)


def is_write(agent: Any, name: str) -> bool:
    return agent._is_write_tool(name) or has_cap(name, EDIT, agent.tool_caps)


def _read_call(agent: Any, idx: int, tc: Any) -> Call:
    tc = as_dict(tc)
    fn = as_dict(tc.get("function", {}))
    args = agent._normalize_arguments(fn.get("arguments") or {})
    # The description is taken out into a COPY, never popped in place:
    # `normalize_arguments` hands back the very dict the assistant message holds, and
    # that message is the record of what the model issued — `take_deferred_calls` reads
    # the calls back out of it to resume a deferred step. Taken out before the dedup key,
    # because two calls differing only in their description are the same call.
    doing = ""
    if isinstance(args, dict) and DOING_ARG in args:
        doing = clip_doing(args[DOING_ARG])
        args = {k: v for k, v in args.items() if k != DOING_ARG}
    call_id = tc.get("id") if isinstance(tc.get("id"), str) and tc.get("id") else f"call_{idx}"
    return Call(fn.get("name", ""), args, call_id, doing)


def admit(tool_calls: list, agent: Any, execution_context: ExecutionContext) -> Admission:
    """Read the step's calls and decide, for each, whether it runs."""
    admission = Admission()
    # (name, args) -> the id of the first call issued that way in this step.
    seen: dict[tuple, str] = {}
    # Per-query loop-control state (dedup + spin guards), outside the schema.
    lc = execution_context.loop_control
    for idx, tc in enumerate(tool_calls):
        call = _read_call(agent, idx, tc)
        if UNPARSED_ARGUMENTS in call.args:
            call.refused = json_error_payload(
                f"The arguments of this {call.name} call are not valid JSON — most likely "
                "the call was cut off at the output limit — so it was not run.",
                hint="Issue the call again, complete. For a large write, split it into "
                     "several smaller calls.",
            )
            emit({"type": "status", "text": f"  ⛔ Not running {call.name}: its arguments were cut off"})
            admission.calls.append(call)
            continue
        key = call.key
        # Not run, yet answered under its own id — left without one, the history repair
        # paired it by position with the next unanswered result, and the model read
        # another call's outcome as its own.
        if key in seen:
            admission.skipped.append((call.call_id, f"Not run: identical to call {seen[key]} "
                                                    "in this step; its result answers both."))
            continue
        seen[key] = call.call_id
        if is_write(agent, call.name):
            # A write already dispatched earlier in this query (after a nudge, say).
            if key in lc.write_calls:
                admission.skipped.append((call.call_id, "Not run: this exact write was "
                                                        "already applied earlier in this query."))
                continue
            lc.write_calls.add(key)
        elif lc.call_fails.get(key, 0) >= HARD_REPEAT_LIMIT:
            # Hard backstop: an identical non-write call that already failed this often
            # gets feedback instead of spinning to the step ceiling.
            call.refused = _repeat_blocked_payload(call.name, lc.call_fails[key])
            emit({"type": "status", "text": f"  ⛔ Blocking repeated failing call: {call.name}"})
        # A run a watcher is already holding answers both of these on its own: asking
        # whether it is done, and spending the turn waiting for it. Refused rather than
        # nudged — the fact is a lookup, not a judgement. Outside the write/non-write
        # branch on purpose (whether a tool writes has nothing to do with whether a run
        # is being watched), and never over a block already decided.
        if call.refused is None:
            watched_key = _asks_whether_a_watched_run_is_done(agent, call.name, call.args)
            if watched_key:
                call.refused = _watched_poll_blocked_payload(watched_key)
                emit({"type": "status", "text":
                      "  ⛔ Already watching that run — you will be resumed when it ends"})
            elif _waits_by_blocking_the_turn(agent, call.name, call.args):
                call.refused = _blocking_wait_payload()
                emit({"type": "status", "text":
                      "  ⛔ Not waiting in-turn — a watcher already holds that run"})
        admission.calls.append(call)
    return admission


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
    for descriptor in _watched_jobs(agent):
        if _op_matches(descriptor.get("summary_op"), name, args):
            return None            # progress, not "are we there yet" — always allowed
        if _op_matches(descriptor.get("status_op"), name, args):
            return str(descriptor.get("job_key") or "?")
    return None


def _op_matches(op: Any, name: str, args: dict) -> bool:
    """True when the call *name(args)* is the descriptor op *op*, by containment."""
    if not isinstance(op, dict) or op.get("tool") != name:
        return False
    op_args = op.get("args")
    if not isinstance(op_args, dict):
        return False
    return all(args.get(k) == v for k, v in op_args.items())


def _reads_progress_of_a_watched_run(agent: Any, name: str, args: dict) -> str | None:
    """The job_key whose output this call reads while a watcher holds it, else None.

    The progress read stays allowed (see above), but it is the one door a poll loop
    can still take: one session read a watched build's output 22 times in 40 seconds,
    its reasoning concluding each time "explain and keep waiting" while the turn
    copied the previous call. The launch note that says to end the turn was a query
    behind by then. The read itself now carries that fact, at the moment of deciding.
    """
    for descriptor in _watched_jobs(agent):
        if _op_matches(descriptor.get("summary_op"), name, args):
            return str(descriptor.get("job_key") or "?")
    return None


def _watched_progress_note(job_key: str) -> str:
    """Appended to a progress read of a watched run."""
    return (
        f"\n\n[background] Job '{job_key}' is still being watched: you will be resumed "
        "automatically with its results when it finishes, and reading its output again "
        "will not bring that sooner. If there is other useful work in this task, carry "
        "on with it now. If the only thing left is waiting for this run, end your turn "
        "and say what you are waiting for."
    )


def _waits_by_blocking_the_turn(agent: Any, name: str, args: dict) -> bool:
    """True when this call's command opens by doing nothing but passing time.

    Registry-driven on both halves: the tool that carries a raw command line is the one
    declaring a shell scope (``capabilities.shell_command_args``), and the program of the
    leading segment comes from the shared segmenter, which skips ``VAR=val`` assignments
    and wrappers of its own. ``sleep`` is named as a POSIX program, the way this
    codebase already names interpreters and wrappers — never as a tool.

    Only the *leading* segment counts. ``./run.sh --sleep 60`` runs a script, and
    ``make; sleep 1`` has already done the work; neither is a turn spent waiting.
    Fail-open on a command the shared parser refuses, as every other shell guard does.
    """
    command_args = shell_command_args(name, getattr(agent, "tool_caps", None))
    if command_args is None:
        return False
    for arg in command_args:
        command = args.get(arg)
        if not isinstance(command, str) or not command.strip():
            continue
        segments = shell_segments(command, allow_expansion=True)
        if not segments:
            continue
        program = segment_program(list(segments[0].argv))
        if program and os.path.basename(program) == "sleep":
            return True
    return False


def _watched_poll_blocked_payload(job_key: str) -> str:
    """Synthetic result for a call asking whether a watched run has finished."""
    return json_error_payload(
        (
            f"Background job '{job_key}' is being watched, so its state was not read. "
            "You will be resumed automatically with its results the moment it finishes."
        ),
        hint=(
            "Do not ask again whether it is done. If there is other useful work in this "
            "task, carry on with it now. If the only thing left is waiting for this run, "
            "end your turn and say what you are waiting for. Reading the run's output "
            "for progress is still available if you need to see how far along it is."
        ),
    )


def _blocking_wait_payload() -> str:
    """Synthetic result for a turn spent waiting on a run something else is watching."""
    return json_error_payload(
        (
            "This command opens by waiting, and a background job is already being "
            "watched for you, so it was not run. Blocking the turn this way also delays "
            "the completion notice it is waiting for, and any message the user sends "
            "meanwhile."
        ),
        hint=(
            "You will be resumed automatically when the run finishes. If there is other "
            "useful work in this task, carry on with it now. If the only thing left is "
            "waiting, end your turn and say what you are waiting for."
        ),
    )


def _repeat_blocked_payload(tool_name: str, fails: int) -> str:
    """Synthetic tool result returned in place of an over-repeated failing call."""
    return json_error_payload(
        (
            f"This exact call failed {fails} times already and was not retried. "
            "Repeating an identical failing call is blocked."
        ),
        hint=(
            "Do not repeat this call. Either change the approach (different arguments, "
            "a different tool, or resolve the underlying environment/precondition), or "
            "stop and conclude clearly that you cannot proceed and why."
        ),
    )

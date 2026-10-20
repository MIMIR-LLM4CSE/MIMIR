"""The mid-loop correctives: reminders that reach the model between two tool steps."""
from __future__ import annotations

import os

from ...config.limits import NUDGE_MAX_TODO_TICK
from ...context.execution_context import nudge_count
from ...guardrails.nudges import inject_reminder, maybe_inject_env_resolution
from ...guardrails.workflow import handback_required, handback_scopes
from ...guardrails.nudges.messages import (
    handback_corrective_message,
    moving_test_corrective_message,
    repeat_corrective_message,
)
from ...context.execution_context import ExecutionContext
from ...core.ports import DispatchAgent


async def post_dispatch_inject(
    agent: DispatchAgent,
    messages: list[dict],
    execution_context: ExecutionContext,
    *,
    active_mode: str = "agent",
) -> None:
    """After a dispatch step, inject the mid-loop reminders the nudge table cannot reach.

    A completed todo step, a repeated identical failure, a moving test, the end of the
    denial ladder, a missing module: the model is still calling tools when they apply.
    """
    # remind agent to mark completed step done in todo list
    success_path = execution_context.edits.last_edit_success_path
    if (
        success_path
        and execution_context.plan.todo_written
        and execution_context.plan.todo_file_path
    ):
        execution_context.edits.last_edit_success_path = ""  # consume
        counts = execution_context.nudges.nudge_counts
        if (
            # The per-category cap every nudge in the table honours, applied here too:
            # this reminder reached the model on EVERY successful write, uncapped, which
            # in a debugging stretch is most of the turn.
            nudge_count(execution_context, "todo_tick") < NUDGE_MAX_TODO_TICK
            # And never twice running about the same file: re-editing what we just spoke
            # about is the middle of one change, not the end of a step.
            and success_path != execution_context.plan.last_todo_tick_path
        ):
            counts["todo_tick"] = counts.get("todo_tick", 0) + 1
            execution_context.plan.last_todo_tick_path = success_path
            inject_reminder(
                messages,
                f"You just wrote {os.path.basename(success_path)} successfully. "
                "If a step in your task checklist is now fully complete, mark it done. "
                "Do NOT mark it done if more work for that step remains.",
                category="todo_tick",
                execution_context=execution_context,
            )

    # one-time corrective for a repeated identical failing call
    alert = execution_context.take("repeat_alert")
    if alert:
        name, fails = alert
        inject_reminder(
            messages, repeat_corrective_message(name, fails), category="repeat_call",
            execution_context=execution_context,
        )

    # one-time corrective for a test file whose failing set keeps changing
    moving_test = execution_context.take("moving_test_alert")
    if moving_test:
        inject_reminder(
            messages, moving_test_corrective_message(moving_test),
            category="moving_test", execution_context=execution_context,
        )

    # one-time stop once refusals reached the end of the denial ladder
    if handback_required(execution_context) and not execution_context.denials.handback_told:
        execution_context.denials.handback_told = True
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

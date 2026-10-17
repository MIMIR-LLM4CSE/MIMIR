"""The repair ladder: what a check or a run that passed or failed does to the workflow.

A file or a run that fails is charged against ``VALIDATION_RETRY_BUDGET`` and sends the
workflow back to ``edit``. Once everything dirty has either passed or used up its
budget, the workflow is released to ``conclude``, so a failure that cannot be fixed is
reported with its residual risk instead of looping.
"""
from __future__ import annotations

from ...servers._shared.shell_paths import EFFECT_BUILD, EFFECT_RUN
from ..context import failed_runs, raise_validation_tier, record_run, run_ledger_key
from ..context.execution_context import ExecutionContext
from .workflow import VALIDATION_RETRY_BUDGET, has_pending_validation, set_workflow_state

BUILD_EXIT_IS_THE_VERDICT = "the build exited 0; a build reports diagnostics, not results"


def mark_file_validated(
    execution_context: ExecutionContext, target: str, tier: str = "static",
) -> None:
    """Credit *target* as checked; conclude when nothing else is pending.

    Only a checker reaches here (``py_compile``, ``ruff``, a compiler): its output is a
    list of problems, so exit 0 means none. Running the code is recorded on the run
    instead (``record_run_outcome``). *tier* (see ``VALIDATION_TIERS``) is read only by
    the completion ledger.
    """
    execution_context.validation_fail_count_by_file.pop(target, None)
    execution_context.validated_files.add(target)
    raise_validation_tier(execution_context, target, tier)
    if execution_context.code_mutation_started and not has_pending_validation(execution_context):
        set_workflow_state(execution_context, "conclude")
        execution_context.nudge_counts["validation"] = 0
        execution_context.nudge_counts["state"] = 0


def register_validation_failure(
    execution_context: ExecutionContext, target: str, tier: str = "static",
) -> None:
    """A check of *target* failed: charge it, and send the workflow back to ``edit``."""
    fail_counts = execution_context.validation_fail_count_by_file
    fail_counts[target] = int(fail_counts.get(target, 0)) + 1
    execution_context.validated_files.discard(target)
    execution_context.validation_tier_by_file.pop(target, None)
    if not execution_context.code_mutation_started:
        return
    set_workflow_state(execution_context, "edit")
    dirty = execution_context.dirty_written_files
    validated = execution_context.validated_files
    all_stuck = dirty and all(
        f in validated or int(fail_counts.get(f, 0)) >= VALIDATION_RETRY_BUDGET
        for f in dirty
    )
    if all_stuck:
        set_workflow_state(execution_context, "conclude")


def record_run_outcome(
    execution_context: ExecutionContext, command: str, completed: bool,
    call_id: str = "", reason: str = "", effect: str = EFFECT_RUN,
    missing_head: str = "", exit_settles: bool = True,
) -> None:
    """Record a run; a run that did not complete is charged to the ladder.

    A green run owes a verdict on what it printed. A green **build** does not: like a
    compiler it reports diagnostics, so it is settled here, provided *exit_settles*
    says the exit status is the build's own (not a pipe's last stage).

    *missing_head* names a command that is not installed: the run is blocked by the
    environment, so it charges nothing.
    """
    run = record_run(execution_context, command, completed=completed, call_id=call_id, effect=effect)
    if completed and effect == EFFECT_BUILD and exit_settles:
        run["verdict"] = "pass"
        run["reason"] = BUILD_EXIT_IS_THE_VERDICT
    if not completed:
        if missing_head:
            run["blocked"] = f"{missing_head} is not installed here"
            execution_context.exercise_blocked_reason = run["blocked"]
            execution_context.exercise_advice_closed = True
        else:
            register_run_failure(execution_context, command, reason)


def register_run_failure(
    execution_context: ExecutionContext, command: str, reason: str = "",
) -> None:
    """A run of *command* failed: charge it, and send the workflow back to ``edit``."""
    run = (execution_context.runs or {}).get(run_ledger_key(command))
    if run is None:
        return
    run["failures"] = int(run.get("failures", 0)) + 1
    if reason:
        run.setdefault("attempts", []).append(reason)
    if not execution_context.code_mutation_started:
        return
    set_workflow_state(execution_context, "edit")
    if all(
        int(r.get("failures", 0)) >= VALIDATION_RETRY_BUDGET
        for r in failed_runs(execution_context).values()
    ) and not has_pending_validation(execution_context):
        set_workflow_state(execution_context, "conclude")

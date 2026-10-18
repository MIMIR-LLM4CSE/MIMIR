"""When each nudge is due: pure reads of the execution_context blackboard.

Verification predicates check reality and apply at every enforcement level; guidance
predicates are a reasoning aid, gated by enforcement and mode through ``tiers``.
"""
from __future__ import annotations

import os
from ...config.limits import VALIDATION_RETRY_BUDGET
from ..workflow import (
    handback_required,
    has_blocking_denials,
    has_pending_validation,
    turn_made_commitments,
    unchecked_checklist_items,
)
from ...context.execution_context import (
    failed_runs,
    known_existing_files,
    nudge_count,
)
from ..builtin_check import builtin_check_failures
from ...config.limits import (
    NUDGE_MAX_BLAST_RADIUS,
    NUDGE_MAX_CREATION,
    NUDGE_MAX_DENIAL,
    NUDGE_MAX_DOC,
    NUDGE_MAX_ENV_CLEANUP,
    NUDGE_MAX_ENV_RESOLUTION,
    NUDGE_MAX_ERROR_RECOVERY,
    NUDGE_MAX_TODO,
    NUDGE_MAX_UNFINISHED_PLAN,
    NUDGE_MAX_UNWATCHED_RUN,
    NUDGE_MAX_VALIDATION,
    STUCK_REPAIR_ADVISE_AFTER,
    STUCK_REPAIR_CONSTRAIN_AFTER,
    TODO_NUDGE_MULTIFILE_THRESHOLD,
    TODO_NUDGE_OP_THRESHOLD,
)
from ...context.execution_context import ExecutionContext
from .tiers import guidance_enabled


def _all_pending_budget_exhausted(execution_context: ExecutionContext) -> bool:
    """True when every dirty file is validated, budget-exhausted, or not checkable at all."""
    dirty = execution_context.edits.dirty_written_files
    if not dirty:
        return False
    validated = execution_context.checks.validated_files
    unverifiable = execution_context.checks.unverifiable_files or set()
    fail_counts = execution_context.checks.validation_fail_count_by_file
    return all(
        f in validated or f in unverifiable
        or int(fail_counts.get(f, 0)) >= VALIDATION_RETRY_BUDGET
        for f in dirty
    )


def has_declared_write_target(execution_context: ExecutionContext) -> bool:
    """True once the model committed, in its own words, to writing something.

    A declared edit target or a recorded plan; reading files is not commitment. The gate
    for nudges that would otherwise push a mutation on a keyword alone.
    """
    if execution_context.edits.planned_edit_targets or execution_context.edits.dirty_written_files:
        return True
    return bool(
        execution_context.plan.todo_written or execution_context.plan.plan_written
    )


def _declared_targets_already_existing(execution_context: ExecutionContext) -> bool:
    """True when a path this turn committed to writing already exists.

    Tells changing (``blast_radius``) from creating (``creation``) by a fact about the
    workspace; reads ``planned_edit_targets`` and ``declared_edit_set``.
    """
    declared = set(execution_context.edits.planned_edit_targets or ()) | set(
        execution_context.edits.declared_edit_set or ()
    )
    if not declared:
        return False
    return bool(declared & known_existing_files(execution_context))


def _retryable_pending_validation_exists(execution_context: ExecutionContext) -> bool:
    """Return True if there is at least one pending file that still has retry budget."""
    dirty = execution_context.edits.dirty_written_files or set()
    validated = execution_context.checks.validated_files or set()
    fail_counts = execution_context.checks.validation_fail_count_by_file
    pending = set(dirty) - set(validated)
    if not pending:
        return False
    return any(int(fail_counts.get(p, 0)) < VALIDATION_RETRY_BUDGET for p in pending)


def untested_edited_sources(execution_context: ExecutionContext) -> list[tuple[str, str]]:
    """``[(source, test)]``: edited Python sources whose ``test_<stem>.py`` exists and did not run."""
    dirty = execution_context.edits.dirty_written_files or set()
    edited_sources = [
        p for p in dirty
        if p.endswith(".py")
        and not (os.path.basename(p).startswith("test_") or os.path.basename(p).endswith("_test.py"))
    ]
    if not edited_sources:
        return []

    known = known_existing_files(execution_context)
    tests_run = execution_context.execution.tests_run or set()
    known_by_base: dict[str, str] = {os.path.basename(p): p for p in known}

    pairs: list[tuple[str, str]] = []
    for src in sorted(edited_sources):
        stem = os.path.basename(src)[:-3]  # strip .py
        for cand in (f"test_{stem}.py", f"{stem}_test.py"):
            test_path = known_by_base.get(cand)
            if test_path and test_path not in tests_run and test_path not in dirty:
                pairs.append((src, test_path))
                break
    return pairs


def _has_non_doc_code_changes(execution_context: ExecutionContext) -> bool:
    dirty = execution_context.edits.dirty_written_files or set()
    if not dirty:
        return False
    return any(not p.endswith(".md") for p in dirty)


def needs_incomplete_finalization(execution_context: ExecutionContext) -> bool:

    # Open steps on the model's own checklist first: validation alone says nothing about
    # steps never started. `turn_made_commitments`, so a turn that declared files and wrote
    # none is not exempt.
    if turn_made_commitments(execution_context) and required_unchecked_steps(
        execution_context
    ):
        return True

    # When the auto-escape fired (all files validated or budget-exhausted),
    # finalize_incomplete_answer will already describe residual risk clearly.
    # Don't add further nudges on top of that.
    if _all_pending_budget_exhausted(execution_context):
        return has_blocking_denials(execution_context)

    # Only the check axis blocks. `workflow_state` is steering, not evidence: reading it
    # here would make the recommended axes mandatory; what a run left open is reported by
    # _collect_completion_issues.
    return (
        has_pending_validation(execution_context)
        or has_blocking_denials(execution_context)
    )


def first_failing_edit_path(execution_context: ExecutionContext) -> str | None:
    """First path with at least one recorded consecutive edit failure, else None.

    Keyed on the per-file failure streak (any patch), so it reflects a model that keeps
    trying differently-anchored edits on the same file, not just identical retries.
    """
    streaks = execution_context.edits.edit_fail_streak_by_file
    for path, fails in streaks.items():
        if fails >= 1:
            return path
    return None


def should_nudge_validation(execution_context: ExecutionContext) -> bool:
    """The built-in check rejected a file, and there is budget left to repair it.

    Verification layer, every level: the one obligation ``needs_incomplete_finalization``
    blocks on. Triggered by a finding, never by a pending file.
    """
    return (
        bool(builtin_check_failures(execution_context))
        and nudge_count(execution_context, "validation") < NUDGE_MAX_VALIDATION
        and _retryable_pending_validation_exists(execution_context)
        and not _all_pending_budget_exhausted(execution_context)
    )


def should_nudge_denial(execution_context: ExecutionContext) -> bool:
    """Blocking denials exist, budget left, and not already in the exhausted dead end.

    The hand-back message escapes the cap and the dead end, but still needs an open denial.
    """
    if handback_required(execution_context) and has_blocking_denials(execution_context):
        return True
    return (
        has_blocking_denials(execution_context)
        and nudge_count(execution_context, "denial") < NUDGE_MAX_DENIAL
        and not _all_pending_budget_exhausted(execution_context)
    )


def should_nudge_error_recovery(execution_context: ExecutionContext) -> bool:
    """Budget left and at least one file is stuck in a repeated edit failure."""
    return (
        nudge_count(execution_context, "error_recovery") < NUDGE_MAX_ERROR_RECOVERY
        and first_failing_edit_path(execution_context) is not None
    )


def worst_run_failure_streak(execution_context: ExecutionContext) -> int:
    """How many times the most-retried failing command has failed (derived, never stored)."""
    return max(
        (int(r.get("failures", 0)) for r in failed_runs(execution_context).values()),
        default=0,
    )


def should_nudge_stuck_repair(execution_context: ExecutionContext) -> bool:
    """One fire per rung, in order (`stuck_repair_nudge_message`).

    The fire count is the rung index, so a streak jumping 2 → 3 never repeats rung one.
    """
    fired = nudge_count(execution_context, "stuck_repair")
    streak = worst_run_failure_streak(execution_context)
    if fired == 0:
        return streak >= STUCK_REPAIR_ADVISE_AFTER
    if fired == 1:
        return streak >= STUCK_REPAIR_CONSTRAIN_AFTER
    return False


def required_unchecked_steps(execution_context: ExecutionContext) -> list[dict]:
    """Unticked checklist steps the plan did not mark optional."""
    return [
        it for it in unchecked_checklist_items(execution_context)
        if not it.get("optional")
    ]


def should_nudge_unfinished_plan(execution_context: ExecutionContext) -> bool:
    """Budget left, the turn committed to something, and its checklist has open steps.

    Verification layer: the evidence is ``- [ ]`` lines the model wrote. False with no
    checklist, and never on a discovery-only turn.
    """
    return (
        nudge_count(execution_context, "unfinished_plan") < NUDGE_MAX_UNFINISHED_PLAN
        and turn_made_commitments(execution_context)
        and bool(required_unchecked_steps(execution_context))
    )


def should_nudge_env_resolution(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A check/run failed for a missing module and envs were not enumerated yet."""
    return (
        guidance_enabled("env_resolution", enforcement=level, active_mode=active_mode)
        and bool(execution_context.env.unresolved_modules)
        and not execution_context.env.env_probed
        and nudge_count(execution_context, "env_resolution") < NUDGE_MAX_ENV_RESOLUTION
    )


def should_nudge_env_cleanup(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """The agent mutated the environment and is now concluding — offer cleanup."""
    return (
        guidance_enabled("env_cleanup", enforcement=level, active_mode=active_mode)
        and bool(execution_context.env.env_mutations)
        and execution_context.workflow_state == "conclude"
        and nudge_count(execution_context, "env_cleanup") < NUDGE_MAX_ENV_CLEANUP
    )


def should_nudge_unwatched_run(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A scheduler listed work of the user's that no record covers, and the turn is concluding.

    The last moment the record can still be written.
    """
    return (
        guidance_enabled("unwatched_run", enforcement=level, active_mode=active_mode)
        and bool(execution_context.execution.unwatched_runs)
        and execution_context.workflow_state == "conclude"
        and nudge_count(execution_context, "unwatched_run") < NUDGE_MAX_UNWATCHED_RUN
    )


def should_nudge_doc(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """Code edited, the workflow at validate/conclude, nothing pending, not all edits docs."""
    return (
        active_mode == "agent"
        and guidance_enabled("doc", enforcement=level, active_mode=active_mode)
        and bool(execution_context.edits.code_mutation_started)
        and execution_context.workflow_state in ("validate", "conclude")
        and not has_pending_validation(execution_context)
        and _has_non_doc_code_changes(execution_context)
        and nudge_count(execution_context, "doc") < NUDGE_MAX_DOC
    )


def should_nudge_blast_radius(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """An existing declared target, nothing written yet, files read, callers never searched."""
    return (
        active_mode == "agent"
        and guidance_enabled("blast_radius", enforcement=level, active_mode=active_mode)
        and execution_context.workflow_state == "edit"
        and not execution_context.edits.code_mutation_started
        and has_declared_write_target(execution_context)
        and _declared_targets_already_existing(execution_context)
        and nudge_count(execution_context, "blast_radius") < NUDGE_MAX_BLAST_RADIUS
        and bool(execution_context.discovery.read_files)
        and int(execution_context.discovery.search_tool_calls) < 1
    )


def should_nudge_creation(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A declared write target that does not exist yet, context gathered, nothing written.

    Exclusive with ``blast_radius``; the message keeps an exit for a question-only turn.
    """
    return (
        active_mode == "agent"
        and guidance_enabled("creation", enforcement=level, active_mode=active_mode)
        and not execution_context.edits.code_mutation_started
        and execution_context.workflow_state in ("edit", "discover")
        and has_declared_write_target(execution_context)
        and not _declared_targets_already_existing(execution_context)
        and bool(execution_context.discovery.read_files)
        and nudge_count(execution_context, "creation") < NUDGE_MAX_CREATION
    )


def should_nudge_todo(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A multi-step task underway with no checklist yet.

    Multi-step: an edit started with ≥ THRESHOLD files touched or planned, or ≥ OP_THRESHOLD
    substantive actions run.
    """
    touched = (
        execution_context.edits.dirty_written_files
        | execution_context.edits.planned_edit_targets
    )
    multi_file = (
        bool(execution_context.edits.code_mutation_started)
        and execution_context.workflow_state == "edit"
        and len(touched) >= TODO_NUDGE_MULTIFILE_THRESHOLD
    )
    many_ops = int(execution_context.action_op_count) >= TODO_NUDGE_OP_THRESHOLD
    return (
        active_mode == "agent"
        and guidance_enabled("todo", enforcement=level, active_mode=active_mode)
        and not execution_context.plan.todo_written
        # A checklist is only useful while work remains. The many-ops trigger has no
        # workflow-state bound, so in a long task that never wrote one it would fire
        # at the very end, after the work is done. Suppressed once concluding: a plan
        # written then guides nothing.
        and execution_context.workflow_state != "conclude"
        and nudge_count(execution_context, "todo") < NUDGE_MAX_TODO
        and (multi_file or many_ops)
    )

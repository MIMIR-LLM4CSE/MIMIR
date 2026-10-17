"""When each nudge is due: pure reads of the execution_context blackboard.

Verification predicates check reality and apply at every enforcement level; guidance
predicates are a reasoning aid, gated by enforcement and mode through ``tiers``.
"""
from __future__ import annotations

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
    declared_edit_set_complete,
    failed_runs,
    idle_steps,
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
    NUDGE_MAX_STATE,
    NUDGE_MAX_TODO,
    NUDGE_MAX_UNFINISHED_PLAN,
    NUDGE_MAX_UNWATCHED_RUN,
    NUDGE_MAX_VALIDATION,
    NUDGE_STATE_IDLE_STEPS,
    STUCK_REPAIR_ADVISE_AFTER,
    STUCK_REPAIR_CONSTRAIN_AFTER,
    TODO_NUDGE_MULTIFILE_THRESHOLD,
    TODO_NUDGE_OP_THRESHOLD,
)
from ...context.execution_context import ExecutionContext
from .tiers import guidance_enabled


def _all_pending_budget_exhausted(execution_context: ExecutionContext) -> bool:
    """Return True when every dirty file is either validated or budget-exhausted.

    In that case the model auto-escaped to conclude and cannot make further
    progress — suppress validation/state nudges to avoid confusing messages.

    A file nothing here can check counts as settled for the same reason: it is
    excluded from ``pending_validation_paths``, so leaving it out of this test made
    one unverifiable file among several validated ones look like outstanding work.
    """
    dirty = execution_context.dirty_written_files
    if not dirty:
        return False
    validated = execution_context.validated_files
    unverifiable = execution_context.unverifiable_files or set()
    fail_counts = execution_context.validation_fail_count_by_file
    return all(
        f in validated or f in unverifiable
        or int(fail_counts.get(f, 0)) >= VALIDATION_RETRY_BUDGET
        for f in dirty
    )


def has_declared_write_target(execution_context: ExecutionContext) -> bool:
    """True once the model has committed, in its own words, to writing something.

    The evidence gate for the two nudges that would otherwise push the model toward a
    mutation on the strength of a *keyword* alone. Reading a file is not commitment —
    an informational question makes the model read files too, which is exactly how a
    query about pip packages ("… on a **new** machine") once produced an unrequested
    script. A declared edit target, or a recorded plan/checklist, is commitment.

    Note the asymmetry with the six state-driven nudges: they observe a mutation that
    already happened (dirty files, failed edits, denials) and so need no such gate.
    """
    for field in ("planned_edit_targets", "dirty_written_files"):
        value = getattr(execution_context, field)
        if isinstance(value, set) and value:
            return True
    return bool(
        execution_context.todo_written or execution_context.plan_written
    )


def _declared_targets_already_existing(execution_context: ExecutionContext) -> bool:
    """True when a path this turn committed to writing is a file that already exists.

    What tells "I am about to change something" from "I am about to create something",
    asked of recorded state. The two nudges that need the distinction — blast-radius
    ("who calls this?") and creation ("you have enough context, write it") — describe
    the same situation otherwise. Separating them by whether the user's query contains an
    edit verb or a create verb is a guess about a request; this is a fact about the
    workspace, which is what such a guess stands in for.

    Both declaration fields are read: ``planned_edit_targets`` (a path already written
    this turn) and ``declared_edit_set`` (paths named in the checklist). A turn that
    committed only to *new* files, or that named no path at all, is not about to break
    a caller — so blast-radius stays silent and creation is free to speak.
    """
    declared = set(execution_context.planned_edit_targets or ()) | set(
        execution_context.declared_edit_set or ()
    )
    if not declared:
        return False
    return bool(declared & known_existing_files(execution_context))


def _retryable_pending_validation_exists(execution_context: ExecutionContext) -> bool:
    """Return True if there is at least one pending file that still has retry budget."""
    dirty = execution_context.dirty_written_files or set()
    validated = execution_context.validated_files or set()
    fail_counts = execution_context.validation_fail_count_by_file
    pending = set(dirty) - set(validated)
    if not pending:
        return False
    return any(int(fail_counts.get(p, 0)) < VALIDATION_RETRY_BUDGET for p in pending)


def untested_edited_sources(execution_context: ExecutionContext) -> list[tuple[str, str]]:
    """Edited Python source files whose associated test exists but was not run this query.

    Returns ``[(source, test)]``. "Associated test" = a known-existing file named
    ``test_<stem>.py`` or ``<stem>_test.py``; "not run" = absent from ``tests_run``.
    Pure reality check (the test file is on disk; it wasn't executed) — model-strength
    independent, so it lives in the verification layer.
    """
    dirty = execution_context.dirty_written_files or set()
    edited_sources = [
        p for p in dirty
        if p.endswith(".py")
        and not (os.path.basename(p).startswith("test_") or os.path.basename(p).endswith("_test.py"))
    ]
    if not edited_sources:
        return []

    known = known_existing_files(execution_context)
    tests_run = execution_context.tests_run or set()
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
    dirty = execution_context.dirty_written_files or set()
    if not dirty:
        return False
    return any(not p.endswith(".md") for p in dirty)


def needs_incomplete_finalization(execution_context: ExecutionContext) -> bool:

    # Open steps on the model's own checklist mean the session is not complete,
    # whatever the validation state says. This runs FIRST, ahead of both validation
    # shortcuts below, because each of them concludes from validation alone: they
    # read "every file I wrote passed a check" (or "no check will ever pass") as
    # "there is nothing left to do". Neither is evidence about steps the model
    # never started — validating the two files it did write says nothing about the
    # three it did not, and unfinished steps are work still available to it even
    # when validation has dead-ended.
    # `turn_made_commitments`, not `code_mutation_started`: the gate is here to spare a
    # discovery-only turn, and "an edit happened" only approximates that. A turn that
    # declared files and wrote none of them is not discovery — it is the case this guard
    # exists for, and the proxy was reading it as exempt.
    if turn_made_commitments(execution_context) and required_unchecked_steps(
        execution_context
    ):
        return True

    # When the auto-escape fired (all files validated or budget-exhausted),
    # finalize_incomplete_answer will already describe residual risk clearly.
    # Don't add further nudges on top of that.
    if _all_pending_budget_exhausted(execution_context):
        return has_blocking_denials(execution_context)

    # Only the check axis blocks. Reading `workflow_state` here as a third condition is
    # what makes the *recommended* axes mandatory in practice: a failed run — or a `fail`
    # verdict, which drives the same ladder — sends the state machine back to `edit`, so
    # every answer comes back "Task is incomplete" until the run has failed
    # VALIDATION_RETRY_BUDGET times. The state machine is
    # steering, not evidence; what a run left open is reported by
    # _collect_completion_issues, which is where a recommendation belongs.
    return (
        has_pending_validation(execution_context)
        or has_blocking_denials(execution_context)
    )


def first_failing_edit_path(execution_context: ExecutionContext) -> str | None:
    """First path with at least one recorded consecutive edit failure, else None.

    Keyed on the per-file failure streak (any patch), so it reflects a model that keeps
    trying differently-anchored edits on the same file, not just identical retries.
    """
    streaks = execution_context.edit_fail_streak_by_file
    for path, fails in streaks.items():
        if fails >= 1:
            return path
    return None


def should_nudge_validation(execution_context: ExecutionContext) -> bool:
    """The built-in check rejected a file, and there is budget left to repair it.

    Verification-layer, at every enforcement level: whether a file one just modified
    parses and is whole is not a reasoning shim a strong model can be trusted out of —
    it is the one obligation of the working order, and the only axis
    ``needs_incomplete_finalization`` blocks on. What is *not* required is anything past
    it: a build needs a toolchain, a run needs an environment, neither is asked here.

    The trigger is a *finding*, not a pending file: the check is performed by the loop
    (``guardrails.builtin_check``) where it asks whether it may conclude, so a file
    still pending has simply not been reached yet and there is nothing to say about it.
    That is also why there are no idle-step or declared-set conditions: they pace a
    request the model is not being asked to carry out.

    Agent-only in practice without a mode test: plan mode is read-only, so nothing is
    dirty and nothing is ever rejected.
    """
    return (
        bool(builtin_check_failures(execution_context))
        and nudge_count(execution_context, "validation") < NUDGE_MAX_VALIDATION
        and _retryable_pending_validation_exists(execution_context)
        and not _all_pending_budget_exhausted(execution_context)
    )


def should_nudge_denial(execution_context: ExecutionContext) -> bool:
    """Blocking denials exist, budget left, and not already in the exhausted dead-end.

    The handback message is exempt from the frequency cap and from the dead-end
    suppression: the other denial messages ask the model to *do* something, so they
    are worth rationing, but this one tells it to stop — a reminder to stop that is
    itself suppressed leaves the model going. It still requires an *open* denial: if
    every refused action was resolved some other way, there is nothing to hand back
    about, and telling a recovered run to stop would cut it short for nothing.
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
    """How many times the most-retried failing command has failed.

    Derived, never stored: ``record_run`` already carries ``failures`` across a re-run,
    so the streak is a read of state that exists. Keeping it a pure function is what
    lets ``nudge_pending`` and ``maybe_append_nudge`` agree — the pre-call probe walks
    the same predicates and must not move any counter.

    Per command rather than summed: two unrelated commands failing once each is not a
    model that is stuck, and adding them would fire on the ordinary red of early work.
    """
    return max(
        (int(r.get("failures", 0)) for r in failed_runs(execution_context).values()),
        default=0,
    )


def should_nudge_stuck_repair(execution_context: ExecutionContext) -> bool:
    """One fire per rung, in order — the ladder in `stuck_repair_nudge_message`.

    Gated on the fire count as a rung *index* rather than on `< NUDGE_MAX_STUCK_REPAIR`:
    a plain cap would let a streak walking 2 → 3 speak the first rung twice, which is the
    repetition this row exists to break. Rung two is therefore never reached without rung
    one having been spoken, whatever the streak jumps to.

    Nothing here resets the count: a `pass` settles the run out of ``failed_runs``, which
    drops the streak to whatever else is still failing, and a query starts with fresh
    counters anyway.
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

    A reality check, not a reasoning shim: the evidence is `- [ ]` lines in a file
    on disk that the model itself wrote. Verification-layer, so it runs at every
    enforcement level — concluding with the plan half-executed is not a mistake a
    stronger model makes less often, and nothing else in the loop reads the
    checklist at completion time.

    Requires a turn that committed to something, so a discovery-only turn is never
    nudged, and degrades to False when there is no checklist at all. That condition used
    to be ``code_mutation_started`` — "an edit happened" — which exempted the very turn
    the nudge is for: one that wrote the plan and then executed none of it.
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
        and bool(execution_context.unresolved_modules)
        and not execution_context.env_probed
        and nudge_count(execution_context, "env_resolution") < NUDGE_MAX_ENV_RESOLUTION
    )


def should_nudge_env_cleanup(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """The agent mutated the environment and is now concluding — offer cleanup."""
    return (
        guidance_enabled("env_cleanup", enforcement=level, active_mode=active_mode)
        and bool(execution_context.env_mutations)
        and execution_context.workflow_state == "conclude"
        and nudge_count(execution_context, "env_cleanup") < NUDGE_MAX_ENV_CLEANUP
    )


def should_nudge_unwatched_run(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A scheduler listed work of the user's that no record covers, and we are done.

    Every condition is a fact somebody checked: the ids came from the queue, the
    absence of a descriptor from the disk, and the clearing from a watcher's handle
    actually returned. Fires at the conclusion because that is the last moment the
    record can still be written — after it, the turn ends, and the run goes unwatched
    for as long as it lasts.
    """
    return (
        guidance_enabled("unwatched_run", enforcement=level, active_mode=active_mode)
        and bool(execution_context.unwatched_runs)
        and execution_context.workflow_state == "conclude"
        and nudge_count(execution_context, "unwatched_run") < NUDGE_MAX_UNWATCHED_RUN
    )


def should_nudge_doc(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """Code changes are done, nothing pending — a gentle documentation reminder.

    Every condition is a recorded fact: a code file was edited, the workflow reached
    validate/conclude, no file still owes a check, and at least one of the edits was not
    itself documentation. What happened says whether a change was made; the wording of
    the request is not consulted.
    """
    return (
        active_mode == "agent"
        and guidance_enabled("doc", enforcement=level, active_mode=active_mode)
        and bool(execution_context.code_mutation_started)
        and execution_context.workflow_state in ("validate", "conclude")
        and not has_pending_validation(execution_context)
        and _has_non_doc_code_changes(execution_context)
        and nudge_count(execution_context, "doc") < NUDGE_MAX_DOC
    )


def should_nudge_state(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """Softer catch-all: editing paused post-validation but not concluded yet."""
    return (
        guidance_enabled("state", enforcement=level, active_mode=active_mode)
        and bool(execution_context.code_mutation_started)
        and execution_context.workflow_state != "conclude"
        and nudge_count(execution_context, "state") < NUDGE_MAX_STATE
        and idle_steps(execution_context) >= NUDGE_STATE_IDLE_STEPS
        and nudge_count(execution_context, "validation") >= 1
        and declared_edit_set_complete(execution_context)
        and not has_pending_validation(execution_context)
        and not _all_pending_budget_exhausted(execution_context)
    )


def should_nudge_blast_radius(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """An EXISTING declared target, nothing written yet, files read, usages never searched.

    Asking about callers is meaningful once the model has named a definition it intends
    to change and has read around it without ever looking for who calls it. That is what
    these conditions say, from recorded state. ``_declared_targets_already_existing``
    tells this row apart from ``creation``: a file that does not exist yet has no callers
    to break.
    """
    return (
        active_mode == "agent"
        and guidance_enabled("blast_radius", enforcement=level, active_mode=active_mode)
        and execution_context.workflow_state == "edit"
        and not execution_context.code_mutation_started
        and has_declared_write_target(execution_context)
        and _declared_targets_already_existing(execution_context)
        and nudge_count(execution_context, "blast_radius") < NUDGE_MAX_BLAST_RADIUS
        and bool(execution_context.read_files)
        and int(execution_context.search_tool_calls) < 1
    )


def should_nudge_creation(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A declared write target that is not an existing file, context gathered, nothing written.

    Reading alone is what answering a question looks like too; a declared write target
    is the model's own recorded intent to create something. The message keeps
    its own exit for the case where the answer was all that was wanted.

    Mutually exclusive with ``blast_radius`` by the same fact that row reads: this one
    owns the turn where nothing named exists yet, so "start writing" cannot land on a
    change that should first have looked for its callers.
    """
    return (
        active_mode == "agent"
        and guidance_enabled("creation", enforcement=level, active_mode=active_mode)
        and not execution_context.code_mutation_started
        and execution_context.workflow_state in ("edit", "discover")
        and has_declared_write_target(execution_context)
        and not _declared_targets_already_existing(execution_context)
        and bool(execution_context.read_files)
        and nudge_count(execution_context, "creation") < NUDGE_MAX_CREATION
    )


def should_nudge_todo(
    execution_context: ExecutionContext, *, level: str, active_mode: str
) -> bool:
    """A multi-step task is underway but no todo list was written yet.

    Two independent triggers, either of which marks the work as multi-step:
      * multi-file: a code edit has started and ≥ THRESHOLD files are touched/planned
        (the classic "editing a set of files" case), or
      * many-ops: ≥ OP_THRESHOLD successful substantive actions (writes/exec/mutations)
        have run — catches tasks that are many operations but few files.
    """
    touched = (
        execution_context.dirty_written_files
        | execution_context.planned_edit_targets
    )
    multi_file = (
        bool(execution_context.code_mutation_started)
        and execution_context.workflow_state == "edit"
        and len(touched) >= TODO_NUDGE_MULTIFILE_THRESHOLD
    )
    many_ops = int(execution_context.action_op_count) >= TODO_NUDGE_OP_THRESHOLD
    return (
        active_mode == "agent"
        and guidance_enabled("todo", enforcement=level, active_mode=active_mode)
        and not execution_context.todo_written
        # A checklist is only useful while work remains. The many-ops trigger has no
        # workflow-state bound, so in a long task that never wrote one it would fire
        # at the very end, after the work is done. Suppressed once concluding: a plan
        # written then guides nothing.
        and execution_context.workflow_state != "conclude"
        and nudge_count(execution_context, "todo") < NUDGE_MAX_TODO
        and (multi_file or many_ops)
    )

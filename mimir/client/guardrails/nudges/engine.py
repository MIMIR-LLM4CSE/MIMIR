
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from ..workflow import (
    VALIDATION_RETRY_BUDGET,
    handback_required,
    has_blocking_denials,
    has_pending_validation,
    unchecked_checklist_items,
)
from .messages import (
    blast_radius_nudge_message,
    creation_nudge_message,
    denial_nudge_message,
    discovery_nudge_message,
    documentation_nudge_message,
    env_cleanup_nudge_message,
    env_resolution_nudge_message,
    error_recovery_nudge_message,
    stuck_repair_nudge_message,
    regression_nudge_message,
    unexercised_code_nudge_message,
    state_nudge_message,
    unfinished_plan_nudge_message,
    todo_nudge_message,
    validation_nudge_message,
)
from ...context.execution_context import (
    backfill_execution_context,
    declared_edit_set_complete,
    failed_runs,
    has_discovery_evidence,
    idle_steps,
    known_existing_files,
    nudge_count,
)
from ....servers._shared.shell_paths import any_command_on_path as _any_command_on_path
from ..builtin_check import builtin_check_failures
from ...config.models import resolve_enforcement
from ...context.capabilities import CODE_EXEC, names_with_cap
from ...event_sink import emit
from ...config.constants import (
    CUSTOM_NUDGE_MAX_PER_QUERY,
    DISCOVERY_EVIDENCE_MIN_DISTINCT,
    EXERCISE_BUDGET,
    NUDGE_MAX_BLAST_RADIUS,
    NUDGE_MAX_CREATION,
    NUDGE_MAX_DENIAL,
    NUDGE_MAX_DISCOVERY,
    NUDGE_MAX_DOC,
    NUDGE_MAX_ENV_CLEANUP,
    NUDGE_MAX_ENV_RESOLUTION,
    NUDGE_MAX_ERROR_RECOVERY,
    NUDGE_MAX_EXERCISE,
    NUDGE_MAX_STATE,
    NUDGE_MAX_TODO,
    NUDGE_MAX_UNFINISHED_PLAN,
    NUDGE_MAX_VALIDATION,
    NUDGE_STATE_IDLE_STEPS,
    STUCK_REPAIR_ADVISE_AFTER,
    STUCK_REPAIR_CONSTRAIN_AFTER,
    TODO_NUDGE_MULTIFILE_THRESHOLD,
    TODO_NUDGE_OP_THRESHOLD,
)
from ...context.signals import (
    query_is_informational,
    query_requires_repo_discovery,
    query_prefers_new_file_creation,
    query_prefers_existing_file_edits,
)
from .plugins import NudgeRegistry, rule_tier_enabled

logger = logging.getLogger(__name__)


def _bootstrap_nudge_context(execution_context: dict[str, Any]) -> dict[str, Any]:
    """Ensure every declared field exists before the nudge table reads it.

    Was a hand-picked list of sixteen ``setdefault`` calls. Besides duplicating three
    similar lists elsewhere, it defaulted ``steps_since_last_edit`` to **99** where the
    schema and every other seeder use 0 — so on a context missing that field the idle
    gates here saw "maximally idle" while ``messages.py`` saw "just edited", and the
    disagreement was absorbed silently by the empty-nudge check. Seeding from the
    schema removes the second opinion.
    """
    return backfill_execution_context(execution_context)


# Prefix marking every workflow nudge as machine-generated rather than user speech.
# Nudges must be injected with role "user" (the Claude backend rejects mid-array
# "system" messages), so without this tag the model cannot tell an automated reminder
# from a real instruction and would let it override the system prompt's latitude clause.
_NUDGE_TAG = "[automated workflow reminder — not from the user; advisory, apply judgment]\n\n"


def inject_reminder(
    messages: list[dict[str, Any]],
    content: str,
    *,
    category: str,
    tagged: bool = True,
) -> None:
    """Append a machine-generated reminder as a user turn and announce the injection.

    EVERY injection point must go through here, not just the nudge table. The event is
    not decoration: the webview holds the turn in flight *outside* the transcript and
    commits it only once the loop accepts it, so a reminder injected silently leaves
    the rejected prose on screen looking like the answer, to be replaced without
    explanation when the real one lands. The loop-control and plan-mode reminders were
    injected that way, which is why the "answer appears then changes" symptom survived
    the draft fix.

    *tagged* is False for the reminders that are protocol, not advice (the plan-mode
    control flow): prefixing those with an "advisory, apply judgment" banner invites
    the model to skip a step the loop actually requires.
    """
    emit({"type": "nudge_injected", "category": category, "text": content})
    messages.append({"role": "user", "content": (_NUDGE_TAG + content) if tagged else content})


# Guidance nudges permitted per (enforcement, mode). This is the single source for
# the enforcement dimension of the line; each nudge's own active_mode/situational
# gate still applies on top. Verification nudges and the plan-mode explore phase are
# independent of this table. "off" is intentionally absent → empty set (and the
# caller short-circuits the whole guidance layer before it is consulted).
# Category strings match the labels passed to _fire_nudge (e.g. the documentation
# nudge's category is "doc", not "documentation").
_ALL_GUIDANCE = frozenset({
    "discovery", "env_resolution", "doc", "state",
    "blast_radius", "creation", "todo", "env_cleanup",
})
_GUIDANCE_BY_LEVEL_MODE: dict[tuple[str, str], frozenset[str]] = {
    # strict babysits everything (the branches still gate mode themselves, so this does
    # not actually leak agent-only nudges into plan mode).
    ("strict", "agent"): _ALL_GUIDANCE,
    ("strict", "plan"): _ALL_GUIDANCE,
    # light: the deliberate carve-out — only the nudges that guard a costly,
    # hard-to-detect, non-self-correcting mistake (blast_radius = breaking callers;
    # env_cleanup = leftover env side effects). Everything else is procedural
    # hand-holding a capable model does unprompted. Absent at "off" (guidance layer
    # short-circuited). Validation is no longer here: checking a file one modified is
    # not a reasoning shim to dial down, it is the one thing the loop requires.
    ("light", "agent"): frozenset({"blast_radius", "env_cleanup"}),
    ("light", "plan"): frozenset(),
    # ask is answer-only: nothing is planned and nothing is edited, so no guidance
    # category has anything to guard. Listed explicitly rather than relying on the
    # missing-key default so the table stays the readable source of truth.
    ("strict", "ask"): frozenset(),
    ("light", "ask"): frozenset(),
}


def _guidance_enabled(category: str, *, enforcement: str, active_mode: str) -> bool:
    """True if *category* may fire at this enforcement level and mode (table lookup).

    Governs ONLY the enforcement dimension — the caller still applies each nudge's
    own ``active_mode``/situational conditions. See ``_GUIDANCE_BY_LEVEL_MODE``.
    """
    return category in _GUIDANCE_BY_LEVEL_MODE.get((enforcement, active_mode), frozenset())


def _fire_nudge(
    execution_context: dict[str, Any],
    messages: list[dict[str, Any]],
    category: str,
    content: str,
    budget_key: str = "",
) -> bool:
    """Increment the nudge counter, append the nudge, and signal it fired.

    Centralises the counter-increment + message-append + ``return True`` triplet that every
    nudge branch in ``maybe_append_nudge`` repeats. The per-nudge cap checks stay in the
    guarding ``if`` so each nudge keeps its own frequency limit. *budget_key* is the
    counter charged when several categories ration one shared budget; the event still
    carries the category, which is what a reader wants to see.
    """
    counts = execution_context["nudge_counts"]
    key = budget_key or category
    counts[key] = counts.get(key, 0) + 1
    logger.debug("nudge fired: category=%s budget=%s count=%d", category, key, counts[key])
    inject_reminder(messages, content, category=category)
    return True


def _all_pending_budget_exhausted(execution_context: dict[str, Any]) -> bool:
    """Return True when every dirty file is either validated or budget-exhausted.

    In that case the model auto-escaped to conclude and cannot make further
    progress — suppress validation/state nudges to avoid confusing messages.

    A file nothing here can check counts as settled for the same reason: it is
    excluded from ``pending_validation_paths``, so leaving it out of this test made
    one unverifiable file among several validated ones look like outstanding work.
    """
    dirty = execution_context.get("dirty_written_files", set())
    if not dirty:
        return False
    validated = execution_context.get("validated_files", set())
    unverifiable = execution_context.get("unverifiable_files", set()) or set()
    fail_counts = execution_context.get("validation_fail_count_by_file", {})
    return all(
        f in validated or f in unverifiable
        or int(fail_counts.get(f, 0)) >= VALIDATION_RETRY_BUDGET
        for f in dirty
    )


def _has_local_discovery_evidence(execution_context: dict[str, Any]) -> bool:
    """True once the model has done real exploration of its own.

    Requires TWO distinct evidence signals so a single stray search/read no longer
    clears the gate. The signal set and the seeded-``inspected_dirs`` exclusion are
    defined once in context.execution_context (has_discovery_evidence).
    """
    return has_discovery_evidence(execution_context, min_distinct=DISCOVERY_EVIDENCE_MIN_DISTINCT)


def _has_declared_write_target(execution_context: dict[str, Any]) -> bool:
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
        value = execution_context.get(field)
        if isinstance(value, set) and value:
            return True
    return bool(
        execution_context.get("todo_written") or execution_context.get("plan_written")
    )


def _retryable_pending_validation_exists(execution_context: dict[str, Any]) -> bool:
    """Return True if there is at least one pending file that still has retry budget."""
    dirty = execution_context.get("dirty_written_files", set()) or set()
    validated = execution_context.get("validated_files", set()) or set()
    fail_counts = execution_context.get("validation_fail_count_by_file", {})
    pending = set(dirty) - set(validated)
    if not pending:
        return False
    return any(int(fail_counts.get(p, 0)) < VALIDATION_RETRY_BUDGET for p in pending)


def _known_existing_files(execution_context: dict[str, Any]) -> set[str]:
    """Files the model has actually encountered this session (proof they exist).

    Thin alias over the shared definition. This module and the policy engine each had
    their own copy of the same four-field walk, with nothing keeping the two equal —
    the drift shape that already cost this codebase a contract test elsewhere.
    """
    return known_existing_files(execution_context)


def _untested_edited_sources(execution_context: dict[str, Any]) -> list[tuple[str, str]]:
    """Edited Python source files whose associated test exists but was not run this query.

    Returns ``[(source, test)]``. "Associated test" = a known-existing file named
    ``test_<stem>.py`` or ``<stem>_test.py``; "not run" = absent from ``tests_run``.
    Pure reality check (the test file is on disk; it wasn't executed) — model-strength
    independent, so it lives in the verification layer.
    """
    dirty = execution_context.get("dirty_written_files", set()) or set()
    edited_sources = [
        p for p in dirty
        if p.endswith(".py")
        and not (os.path.basename(p).startswith("test_") or os.path.basename(p).endswith("_test.py"))
    ]
    if not edited_sources:
        return []

    known = _known_existing_files(execution_context)
    tests_run = execution_context.get("tests_run", set()) or set()
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


def _has_non_doc_code_changes(execution_context: dict[str, Any]) -> bool:
    dirty = execution_context.get("dirty_written_files", set()) or set()
    if not dirty:
        return False
    return any(not p.endswith(".md") for p in dirty)


def needs_incomplete_finalization(execution_context: dict[str, Any]) -> bool:
    execution_context = _bootstrap_nudge_context(execution_context)

    # Open steps on the model's own checklist mean the session is not complete,
    # whatever the validation state says. This runs FIRST, ahead of both validation
    # shortcuts below, because each of them concludes from validation alone: they
    # read "every file I wrote passed a check" (or "no check will ever pass") as
    # "there is nothing left to do". Neither is evidence about steps the model
    # never started — validating the two files it did write says nothing about the
    # three it did not, and unfinished steps are work still available to it even
    # when validation has dead-ended.
    if execution_context.get("code_mutation_started") and _required_unchecked_steps(
        execution_context
    ):
        return True

    # When the auto-escape fired (all files validated or budget-exhausted),
    # finalize_incomplete_answer will already describe residual risk clearly.
    # Don't add further nudges on top of that.
    if _all_pending_budget_exhausted(execution_context):
        return has_blocking_denials(execution_context)

    # Only the check axis blocks. `workflow_state` used to be read here as a third
    # condition, and that is what made the *recommended* axes mandatory in practice:
    # a failed run — or a `fail` verdict, which drives the same ladder — sends the
    # state machine back to `edit`, so every answer came back "Task is incomplete"
    # until the run had failed VALIDATION_RETRY_BUDGET times. The state machine is
    # steering, not evidence; what a run left open is reported by
    # _collect_completion_issues, which is where a recommendation belongs.
    return (
        has_pending_validation(execution_context)
        or has_blocking_denials(execution_context)
    )


def maybe_append_nudge(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    execution_context: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bool:
    """Append at most one workflow nudge, in priority order.

    Built-in nudges are an ordered table (``_CORE_NUDGES``) walked by the generic
    ``_append_core_nudge`` runner; application packs add more via the ``NudgeRegistry``
    (``_append_custom_nudge``). Both share the same shape (name + layer + predicate +
    render). Two clearly separated layers:

    - **Verification reminders** (``layer="verification"``) check *reality* — denied
      required actions, repeated edit failures, an edited source whose test was never
      run, a checklist still open after code was written. They run regardless of model
      strength because a smarter model is no more honest about verification and has no
      ground-truth access to disk/process state.
    - **Guidance nudges** (``layer="guidance"``) babysit the model's *reasoning*
      (validation, env resolution/cleanup, discovery, doc, state, blast-radius,
      creation, todo). They
      are a weak-model compatibility shim, skipped entirely when enforcement is
      ``"off"``; which subset survives at ``"light"`` is defined by
      ``_GUIDANCE_BY_LEVEL_MODE`` (consulted inside each guidance predicate).

    Order per layer: core table first, then packs. Verification is tried before
    guidance, so it always wins.
    """
    execution_context = _bootstrap_nudge_context(execution_context)
    level = resolve_enforcement(agent)

    # Verification reminders run regardless of model strength.
    if _append_core_nudge(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, messages=messages,
        layer="verification", level=level,
    ):
        return True

    # Application verification nudges (extension packs) — always on, like core
    # verification. Run after core so a built-in reality check always wins.
    if _append_custom_nudge(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, messages=messages, layer="verification",
    ):
        return True

    # Guidance nudges are a weak-model compatibility shim — skip entirely when
    # the model's enforcement level is "off". This gate applies to core AND
    # application guidance nudges alike.
    if level == "off":
        _log_no_nudge(query, execution_context, level=level, active_mode=active_mode)
        return False

    if _append_core_nudge(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, messages=messages,
        layer="guidance", level=level,
    ):
        return True

    # Application guidance nudges (extension packs) — tier-gated, after core guidance.
    fired = _append_custom_nudge(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, messages=messages, layer="guidance",
    )
    if not fired:
        _log_no_nudge(query, execution_context, level=level, active_mode=active_mode)
    return fired


def nudge_pending(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    execution_context: dict[str, Any],
) -> bool:
    """True if :func:`maybe_append_nudge` would fire right now — predicate-only probe.

    Read BEFORE the model call so the loop can hold the turn's prose instead of
    streaming it: a turn a nudge sends back must never have reached the screen.
    Walks the same table and the same registry in the same layer order, evaluating
    only the predicates — nothing is rendered, injected or counted, so probing has
    no side effect on the run.

    Deliberately an over-approximation on one point: a row whose predicate matches
    but whose render comes back empty is reported as pending. The cost is one turn
    that reaches the screen late; the alternative is rendering twice.
    """
    execution_context = _bootstrap_nudge_context(execution_context)
    level = resolve_enforcement(agent)
    if _core_nudge_pending(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, layer="verification", level=level,
    ) or _custom_nudge_pending(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, layer="verification",
    ):
        return True
    if level == "off":
        return False
    return _core_nudge_pending(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, layer="guidance", level=level,
    ) or _custom_nudge_pending(
        agent=agent, query=query, active_mode=active_mode,
        execution_context=execution_context, layer="guidance",
    )


def _log_no_nudge(
    query: str,
    execution_context: dict[str, Any],
    *,
    level: str,
    active_mode: str,
) -> None:
    """Record why no nudge was injected on a turn that asked for one.

    The counterpart to the fire-time log in :func:`_fire_nudge`. Suppression used to be
    invisible: a mis-classified query silently disarmed the guidance layer with nothing
    in the trace to explain it. Debug-level and lazily formatted, so it costs nothing
    when the logger is off; the intent/veto booleans are what make a false negative
    diagnosable after the fact.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "no nudge: level=%s mode=%s informational=%s create=%s edit=%s "
        "declared_target=%s counts=%s",
        level,
        active_mode,
        query_is_informational(query),
        query_prefers_new_file_creation(query),
        query_prefers_existing_file_edits(query),
        _has_declared_write_target(execution_context),
        execution_context.get("nudge_counts", {}),
    )


def _append_custom_nudge(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    execution_context: dict[str, Any],
    messages: list[dict[str, Any]],
    layer: str,
) -> bool:
    """Fire at most one application-registered nudge of *layer*, in priority order.

    Preserves every core invariant: toggle suppression (``agent.disabled_nudges``),
    tier gating for guidance rules (``rule_tier_enabled``), the per-query frequency
    cap, and injection through the shared ``_fire_nudge`` (``_NUDGE_TAG`` + role
    ``"user"`` + counter). A rule whose predicate/render raises is skipped, so a bad
    pack cannot break the loop.
    """
    disabled = getattr(agent, "disabled_nudges", None) or set()
    counts = execution_context["nudge_counts"]
    enforcement = resolve_enforcement(agent)
    for rule in NudgeRegistry.rules(layer):
        if rule.name in disabled:
            continue
        if counts.get(rule.name, 0) >= CUSTOM_NUDGE_MAX_PER_QUERY:
            continue
        if layer == "guidance" and not rule_tier_enabled(
            rule, enforcement=enforcement, active_mode=active_mode
        ):
            continue
        try:
            if not rule.predicate(agent, query, active_mode, execution_context):
                continue
            content = rule.render(agent, execution_context)
        except Exception:
            continue
        if not content or not str(content).strip():
            continue
        return _fire_nudge(execution_context, messages, rule.name, str(content))
    return False


def _custom_nudge_pending(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    execution_context: dict[str, Any],
    layer: str,
) -> bool:
    """Would any pack-registered rule of *layer* fire? Mirrors the gates of
    :func:`_append_custom_nudge` (toggle, per-query cap, tier) without rendering."""
    disabled = getattr(agent, "disabled_nudges", None) or set()
    counts = execution_context["nudge_counts"]
    enforcement = resolve_enforcement(agent)
    for rule in NudgeRegistry.rules(layer):
        if rule.name in disabled:
            continue
        if counts.get(rule.name, 0) >= CUSTOM_NUDGE_MAX_PER_QUERY:
            continue
        if layer == "guidance" and not rule_tier_enabled(
            rule, enforcement=enforcement, active_mode=active_mode
        ):
            continue
        try:
            if rule.predicate(agent, query, active_mode, execution_context):
                return True
        except Exception:
            continue
    return False


def _first_failing_edit_path(execution_context: dict[str, Any]) -> str | None:
    """First path with at least one recorded consecutive edit failure, else None.

    Keyed on the per-file failure streak (any patch), so it reflects a model that keeps
    trying differently-anchored edits on the same file, not just identical retries.
    """
    streaks = execution_context.get("edit_fail_streak_by_file", {})
    for path, fails in streaks.items():
        if fails >= 1:
            return path
    return None


# ── Verification-layer predicates (reality checks; run at every enforcement level) ──

def _should_nudge_validation(
    execution_context: dict[str, Any], *, level: str, active_mode: str,
) -> bool:
    """The built-in check rejected a file, and there is budget left to repair it.

    Verification-layer, at every enforcement level: whether a file one just modified
    parses and is whole is not a reasoning shim a strong model can be trusted out of —
    it is the one obligation of the working order, and the only axis
    ``needs_incomplete_finalization`` blocks on. What is *not* required is anything past
    it: a build needs a toolchain, a run needs an environment, neither is asked here.

    The trigger is a *finding*, not a pending file: the check is performed by the loop
    (``guardrails.builtin_check``) where it asks whether it may conclude, so a file
    still pending has simply not been reached yet and there is nothing to say about it.
    That is also why the idle-step and declared-set conditions are gone — they paced a
    request the model no longer has to carry out.

    Agent-only in practice without a mode test: plan mode is read-only, so nothing is
    dirty and nothing is ever rejected.
    """
    return (
        bool(builtin_check_failures(execution_context))
        and nudge_count(execution_context, "validation") < NUDGE_MAX_VALIDATION
        and _retryable_pending_validation_exists(execution_context)
        and not _all_pending_budget_exhausted(execution_context)
    )


def _validation_nudge_content(agent: Any, execution_context: dict[str, Any]) -> str:
    # Short pointer only — the full TIER 1 rationale already lives in the system
    # prompt; re-asserting it here as a user-role message would just duplicate the
    # strictest content through the more coercive channel. Pending validation is
    # handled uniformly regardless of whether the modified files are sources or tests.
    return validation_nudge_message(agent, execution_context)


def _should_nudge_denial(execution_context: dict[str, Any]) -> bool:
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


def _should_nudge_error_recovery(execution_context: dict[str, Any]) -> bool:
    """Budget left and at least one file is stuck in a repeated edit failure."""
    return (
        nudge_count(execution_context, "error_recovery") < NUDGE_MAX_ERROR_RECOVERY
        and _first_failing_edit_path(execution_context) is not None
    )


def _worst_run_failure_streak(execution_context: dict[str, Any]) -> int:
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


def _should_nudge_stuck_repair(execution_context: dict[str, Any]) -> bool:
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
    streak = _worst_run_failure_streak(execution_context)
    if fired == 0:
        return streak >= STUCK_REPAIR_ADVISE_AFTER
    if fired == 1:
        return streak >= STUCK_REPAIR_CONSTRAIN_AFTER
    return False


# ── The advisory axis: build it, run it ───────────────────────────────────
# `regression` and `unexercised` are two phrasings of one question — "does anything
# actually show this works?" — so they ration ONE budget rather than two, and asking it
# is a recommendation: a build needs a toolchain and a run needs an environment, neither
# of which the loop can conjure. Once the model has answered (a run it judged, or an
# `unknown` saying it cannot), the subject is closed for the rest of the query. The
# shared counter key is `EXERCISE_BUDGET` (config.constants).
#
# There is deliberately no third row asking for a *verdict* on a run that has one
# outstanding. That ask fired on the happy path — edit, run, all green, answer — and
# bought a label the ledger already prints ("its output was never judged") with a whole
# discarded final answer plus, usually, a re-run to recover output the model no longer
# had in front of it. The verdict is asked for where it costs nothing: the VERDICT_DUE
# annotation on the run's own result, and the judging tool's docstring.

# What this box can start with one direct command, by edited-file kind. The value is the
# runner that has to exist for that to be true — asked of PATH, never assumed, so a `.py`
# edit on a box with no interpreter is correctly not a route.
_DIRECT_RUNNERS: dict[str, tuple[str, ...]] = {
    ".py": ("python3", "python"),
    ".sh": ("sh",),
}

# A suite already registered against a configured tree: `ctest` is one direct command, and
# unlike a build it produces a result somebody can judge. The file is generated at configure
# time and only when tests exist, so its presence is the whole test — a compiled test
# *source* is deliberately not a route, because reaching it still means building first.
_REGISTERED_SUITES: dict[str, tuple[str, ...]] = {
    "CTestTestfile.cmake": ("ctest",),
}

# A build only counts as a route when the tree is ALREADY configured — that is the whole
# proportionality test. `CMakeCache.txt` and not `CMakeLists.txt`: a configured tree is one
# direct command, configuring one is a step of its own and stays out of reach.
_CONFIGURED_BUILDS: dict[str, tuple[str, ...]] = {
    "Makefile": ("make",),
    "makefile": ("make",),
    "GNUmakefile": ("make",),
    "CMakeCache.txt": ("cmake", "make"),
}


def _exercise_budget_left(execution_context: dict[str, Any]) -> bool:
    """True while the one shared exercise/verdict ask is still available."""
    return (
        not execution_context.get("exercise_advice_closed")
        and nudge_count(execution_context, EXERCISE_BUDGET) < NUDGE_MAX_EXERCISE
    )


def _exercise_route(agent: Any, execution_context: dict[str, Any]) -> str:
    """The one direct command that would exercise this change, named — or "" if none.

    This is where "recommended when it is simply feasible" is decided, rather than
    recited. A route is one invocation against the project as it stands, using something
    that is actually installed; anything needing a step of its own first — configuring a
    build, installing a package, provisioning data — is out of proportion and returns "".
    So the model is never pushed toward a disproportionate step, and a recommendation that
    does fire can name the command it found.

    Silent, but not invisible: the obstacle is recorded so the ledger can say what stood in
    the way instead of simply omitting the advice. Evidence only — a build file has to have
    been *seen* this session, because recommending `make` against a Makefile nobody has laid
    eyes on is not obviously proportionate either.
    """
    def _blocked(reason: str) -> str:
        execution_context["exercise_blocked_reason"] = reason
        return ""

    def _route(description: str) -> str:
        execution_context["exercise_blocked_reason"] = ""
        return description

    if execution_context.get("unresolved_modules"):
        return _blocked("an import could not be resolved in the interpreter that was used")
    if not names_with_cap(CODE_EXEC, getattr(agent, "tool_caps", None)):
        return _blocked("no execution tool is connected to this session")

    # 1. A test that already covers the edit: the cheapest route there is, and the one the
    #    regression row is built on.
    untested = _untested_edited_sources(execution_context)
    if untested:
        return _route(f"the test that already covers it ({untested[0][1]})")

    dirty = sorted(execution_context.get("dirty_written_files") or ())
    known = {os.path.basename(p) for p in _known_existing_files(execution_context)}

    # 2. A file this environment starts directly.
    for path in dirty:
        runners = _DIRECT_RUNNERS.get(os.path.splitext(path)[1].lower())
        if runners and _any_command_on_path(runners):
            return _route(f"running {path} directly")

    needs_build = [p for p in dirty if os.path.splitext(p)[1].lower() not in _DIRECT_RUNNERS]
    if not needs_build:
        return _blocked("nothing written here starts with one direct command")

    # 3. A suite already registered here — ahead of the plain build below, and for the
    #    reason the tier ladder gives: a build says the code is well formed, only a run
    #    produces a result. This is what route 1 is for a compiled language, where pairing
    #    a test *file* to a source would name something that still has to be built.
    for suite_file, runners in _REGISTERED_SUITES.items():
        if suite_file in known and _any_command_on_path(runners):
            return _route(f"the test suite already registered here ({runners[0]})")

    # 4. A build already configured in the tree — the branch a suffix test used to refuse
    #    by category, which left every compiled change with no recommendation at all.
    for build_file, drivers in _CONFIGURED_BUILDS.items():
        if build_file in known and _any_command_on_path(drivers):
            return _route(f"the build already configured here ({build_file})")

    return _blocked("nothing written here starts with one direct command")


def _should_nudge_regression(execution_context: dict[str, Any]) -> bool:
    """Budget left and an edited source has an on-disk test that was never run."""
    return (
        _exercise_budget_left(execution_context)
        and bool(_untested_edited_sources(execution_context))
    )


def _required_unchecked_steps(execution_context: dict[str, Any]) -> list[dict]:
    """Unticked checklist steps the plan did not mark optional."""
    return [
        it for it in unchecked_checklist_items(execution_context)
        if not it.get("optional")
    ]


def _should_nudge_unfinished_plan(execution_context: dict[str, Any]) -> bool:
    """Budget left, code was written, and the model's own checklist has open steps.

    A reality check, not a reasoning shim: the evidence is `- [ ]` lines in a file
    on disk that the model itself wrote. Verification-layer, so it runs at every
    enforcement level — concluding with the plan half-executed is not a mistake a
    stronger model makes less often, and nothing else in the loop reads the
    checklist at completion time.

    Requires ``code_mutation_started`` so a discovery-only turn is never nudged,
    and degrades to False when there is no checklist at all.
    """
    return (
        nudge_count(execution_context, "unfinished_plan") < NUDGE_MAX_UNFINISHED_PLAN
        and bool(execution_context.get("code_mutation_started"))
        and bool(_required_unchecked_steps(execution_context))
    )


def _should_nudge_unexercised_code(agent: Any, execution_context: dict[str, Any]) -> bool:
    """Shared budget left, every written file checked, and nothing was ever run.

    Mutually exclusive with the two rows around it: ``validation`` speaks while a file
    still owes a check, this one owns the state after — checks all green, no run at all
    — which is exactly where a model stops believing it is finished. A checker
    established that the file is written correctly; nothing yet bears on whether it
    computes the right thing, and nothing will until something executes.

    A recommendation, not a requirement, which is why it is bounded three ways: the
    shared exercise budget, the feasibility gate, and the fact that "there was nothing
    to run" is an answer the message explicitly accepts. Nothing downstream blocks on
    it — an unexercised change concludes, and the ledger says so.
    """
    return (
        _exercise_budget_left(execution_context)
        and bool(execution_context.get("code_mutation_started"))
        and bool(execution_context.get("dirty_written_files"))
        and not has_pending_validation(execution_context)
        and not (execution_context.get("runs") or {})
        and bool(_exercise_route(agent, execution_context))
    )


# ── Guidance-layer predicates (reasoning shim; gated by enforcement + mode) ──

def _should_nudge_env_resolution(
    execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """A check/run failed for a missing module and envs were not enumerated yet."""
    return (
        _guidance_enabled("env_resolution", enforcement=level, active_mode=active_mode)
        and bool(execution_context.get("unresolved_modules"))
        and not execution_context.get("env_probed")
        and nudge_count(execution_context, "env_resolution") < NUDGE_MAX_ENV_RESOLUTION
    )


def maybe_inject_env_resolution(
    *,
    agent: Any,
    active_mode: str,
    execution_context: dict[str, Any],
    messages: list[dict[str, Any]],
) -> bool:
    """Fire the env-resolution cascade mid-loop, right after the call that failed.

    Every other nudge answers "is the work done?", which is a question worth asking
    once the model stops calling tools. This one answers "why did that just fail?", and
    a step ceiling away from the failure the answer is worth much less: the model has
    already spent its steps retrying against the same interpreter, and the generic
    repeat guard only catches it when the retries are *identical*.

    Charged to the same counter as the table row it duplicates, so whichever fires
    first spends the single budget and the other stays silent. Same enforcement/mode
    gate too — this changes *when* the cascade arrives, not *whether* it applies.
    """
    execution_context = _bootstrap_nudge_context(execution_context)
    if "env_resolution" in (getattr(agent, "disabled_nudges", None) or set()):
        return False
    level = resolve_enforcement(agent)
    if level == "off":
        return False
    if not _should_nudge_env_resolution(
        execution_context, level=level, active_mode=active_mode
    ):
        return False
    return _fire_nudge(
        execution_context, messages, "env_resolution",
        env_resolution_nudge_message(execution_context),
    )


def _should_nudge_env_cleanup(
    execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """The agent mutated the environment and is now concluding — offer cleanup."""
    return (
        _guidance_enabled("env_cleanup", enforcement=level, active_mode=active_mode)
        and bool(execution_context.get("env_mutations"))
        and execution_context.get("workflow_state") == "conclude"
        and nudge_count(execution_context, "env_cleanup") < NUDGE_MAX_ENV_CLEANUP
    )


def _should_nudge_discovery(
    query: str, execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """Repo discovery is expected but the model has almost no local evidence yet."""
    return (
        active_mode == "agent"
        and _guidance_enabled("discovery", enforcement=level, active_mode=active_mode)
        and query_requires_repo_discovery(query)
        and nudge_count(execution_context, "discovery") < NUDGE_MAX_DISCOVERY
        and not _has_local_discovery_evidence(execution_context)
    )


def _should_nudge_doc(
    query: str, execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """Code changes are done, nothing pending — a gentle documentation reminder."""
    return (
        active_mode == "agent"
        and _guidance_enabled("doc", enforcement=level, active_mode=active_mode)
        and bool(execution_context.get("code_mutation_started"))
        and execution_context.get("workflow_state") in ("validate", "conclude")
        and not has_pending_validation(execution_context)
        and _has_non_doc_code_changes(execution_context)
        and (query_prefers_new_file_creation(query) or query_prefers_existing_file_edits(query))
        and nudge_count(execution_context, "doc") < NUDGE_MAX_DOC
    )


def _should_nudge_state(
    execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """Softer catch-all: editing paused post-validation but not concluded yet."""
    return (
        _guidance_enabled("state", enforcement=level, active_mode=active_mode)
        and bool(execution_context.get("code_mutation_started"))
        and execution_context.get("workflow_state") != "conclude"
        and nudge_count(execution_context, "state") < NUDGE_MAX_STATE
        and idle_steps(execution_context) >= NUDGE_STATE_IDLE_STEPS
        and nudge_count(execution_context, "validation") >= 1
        and declared_edit_set_complete(execution_context)
        and not has_pending_validation(execution_context)
        and not _all_pending_budget_exhausted(execution_context)
    )


def _should_nudge_blast_radius(
    query: str, execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """Edit intent, a declared target, nothing written yet, usages not searched.

    Asking about callers is only meaningful once the model has named the definition it
    intends to change — hence the declared-target gate on top of the keyword intent.
    """
    return (
        active_mode == "agent"
        and _guidance_enabled("blast_radius", enforcement=level, active_mode=active_mode)
        and query_prefers_existing_file_edits(query)
        and execution_context.get("workflow_state") == "edit"
        and not execution_context.get("code_mutation_started")
        and _has_declared_write_target(execution_context)
        and nudge_count(execution_context, "blast_radius") < NUDGE_MAX_BLAST_RADIUS
        and bool(execution_context.get("read_files"))
        and int(execution_context.get("search_tool_calls", 0)) < 1
    )


def _should_nudge_creation(
    query: str, execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """Create intent, a declared write target, context gathered, still nothing written.

    ``read_files`` alone used to stand in for "the model is mid-task", but reading is
    what answering a question looks like too. The declared-target gate is what keeps a
    keyword match from turning an answer into an unrequested file.
    """
    return (
        active_mode == "agent"
        and _guidance_enabled("creation", enforcement=level, active_mode=active_mode)
        and query_prefers_new_file_creation(query)
        and not execution_context.get("code_mutation_started")
        and execution_context.get("workflow_state") in ("edit", "discover")
        and _has_declared_write_target(execution_context)
        and bool(execution_context.get("read_files"))
        and nudge_count(execution_context, "creation") < NUDGE_MAX_CREATION
    )


def _should_nudge_todo(
    execution_context: dict[str, Any], *, level: str, active_mode: str
) -> bool:
    """A multi-step task is underway but no todo list was written yet.

    Two independent triggers, either of which marks the work as multi-step:
      * multi-file: a code edit has started and ≥ THRESHOLD files are touched/planned
        (the classic "editing a set of files" case), or
      * many-ops: ≥ OP_THRESHOLD successful substantive actions (writes/exec/mutations)
        have run — catches tasks that are many operations but few files.
    """
    touched = (
        execution_context.get("dirty_written_files", set())
        | execution_context.get("planned_edit_targets", set())
    )
    multi_file = (
        bool(execution_context.get("code_mutation_started"))
        and execution_context.get("workflow_state") == "edit"
        and len(touched) >= TODO_NUDGE_MULTIFILE_THRESHOLD
    )
    many_ops = int(execution_context.get("action_op_count", 0)) >= TODO_NUDGE_OP_THRESHOLD
    return (
        active_mode == "agent"
        and _guidance_enabled("todo", enforcement=level, active_mode=active_mode)
        and not execution_context.get("todo_written", False)
        # A checklist is only useful while work remains. The many-ops trigger has no
        # workflow-state bound, so in a long task that never wrote one it would fire
        # at the very end — the model then writes a plan/todo AFTER the work is done
        # (observed: proxy optimization emitting a todo list at conclusion). Suppress
        # once concluding: a plan written then guides nothing.
        and execution_context.get("workflow_state") != "conclude"
        and nudge_count(execution_context, "todo") < NUDGE_MAX_TODO
        and (multi_file or many_ops)
    )


# ── Core nudge table + generic runner ──────────────────────────────────────────
# The built-in nudges declared as an ordered table, mirroring the pack-facing
# NudgeRule shape (name + layer + predicate + render). Verification rows are reality
# checks that run at every enforcement level; guidance rows carry their own
# enforcement/mode gate inside their predicate (via _guidance_enabled). Ordering IS
# priority — the runner fires the first matching row of the requested layer. The
# per-category frequency cap also lives inside each predicate (nudge_count < NUDGE_MAX_*).
# Adding a built-in nudge is a one-row change plus its _should_nudge_* predicate.

@dataclass(frozen=True)
class _CoreNudge:
    name: str        # nudge_counts category / toggle key
    layer: str       # "verification" | "guidance"
    should_fire: Callable[[Any, str, str, dict[str, Any], str], bool]  # (agent, query, mode, ec, level)
    render: Callable[[Any, dict[str, Any]], str]                       # (agent, ec) -> text
    budget_key: str = ""  # counter charged when rows ration a shared budget; default = name


_CORE_NUDGES: tuple[_CoreNudge, ...] = (
    # ── Verification (reality checks; always run, independent of enforcement) ──
    _CoreNudge(
        "denial", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_denial(ec),
        lambda agent, ec: denial_nudge_message(ec),
    ),
    # The client already attaches a structured failure diagnosis automatically, so
    # the model does not need to request one itself — just steer it to re-read.
    _CoreNudge(
        "error_recovery", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_error_recovery(ec),
        lambda agent, ec: error_recovery_nudge_message(_first_failing_edit_path(ec)),
    ),
    # Ahead of the three advisory rows below, because it is the only one of the four
    # that is required: a file the model modified and never checked is the one gap
    # ``needs_incomplete_finalization`` refuses to conclude over.
    # Before `validation`, and deliberately: while the model is going round the same
    # failure, telling it to finish checking is answering a question it is not stuck on.
    # Only one row speaks per turn, so this suppresses `validation` for as long as the
    # ladder is talking — which costs nothing, because pending validation still blocks
    # the conclusion through `needs_incomplete_finalization` either way.
    _CoreNudge(
        "stuck_repair", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_stuck_repair(ec),
        lambda agent, ec: stuck_repair_nudge_message(_worst_run_failure_streak(ec)),
    ),
    _CoreNudge(
        "validation", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_validation(ec, level=level, active_mode=mode),
        lambda agent, ec: _validation_nudge_content(agent, ec),
    ),
    _CoreNudge(
        "regression", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_regression(ec),
        lambda agent, ec: regression_nudge_message(_untested_edited_sources(ec)),
        budget_key=EXERCISE_BUDGET,
    ),
    # Between them by design: `regression` covers an edit whose tests already exist,
    # this one the change that has no run at all behind it.
    _CoreNudge(
        "unexercised", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_unexercised_code(agent, ec),
        lambda agent, ec: unexercised_code_nudge_message(
            sorted(ec.get("dirty_written_files") or ()), _exercise_route(agent, ec)),
        budget_key=EXERCISE_BUDGET,
    ),
    _CoreNudge(
        "unfinished_plan", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_unfinished_plan(ec),
        lambda agent, ec: unfinished_plan_nudge_message(_required_unchecked_steps(ec)),
    ),
    # ── Guidance (reasoning shim; each predicate owns its enforcement/mode gate) ──
    _CoreNudge(
        "env_resolution", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_env_resolution(ec, level=level, active_mode=mode),
        lambda agent, ec: env_resolution_nudge_message(ec),
    ),
    _CoreNudge(
        "env_cleanup", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_env_cleanup(ec, level=level, active_mode=mode),
        lambda agent, ec: env_cleanup_nudge_message(ec),
    ),
    _CoreNudge(
        "discovery", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_discovery(query, ec, level=level, active_mode=mode),
        lambda agent, ec: discovery_nudge_message(),
    ),
    _CoreNudge(
        "doc", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_doc(query, ec, level=level, active_mode=mode),
        lambda agent, ec: documentation_nudge_message(ec),
    ),
    _CoreNudge(
        "state", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_state(ec, level=level, active_mode=mode),
        lambda agent, ec: state_nudge_message(agent, ec),
    ),
    _CoreNudge(
        "blast_radius", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_blast_radius(query, ec, level=level, active_mode=mode),
        lambda agent, ec: blast_radius_nudge_message(ec),
    ),
    _CoreNudge(
        "creation", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_creation(query, ec, level=level, active_mode=mode),
        lambda agent, ec: creation_nudge_message(ec),
    ),
    _CoreNudge(
        "todo", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_todo(ec, level=level, active_mode=mode),
        lambda agent, ec: todo_nudge_message(ec),
    ),
)


def _core_nudge_pending(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    execution_context: dict[str, Any],
    layer: str,
    level: str,
) -> bool:
    """Would any built-in row of *layer* fire? Same table, predicates only."""
    return any(
        spec.layer == layer
        and spec.should_fire(agent, query, active_mode, execution_context, level)
        for spec in _CORE_NUDGES
    )


def _append_core_nudge(
    *,
    agent: Any,
    query: str,
    active_mode: str,
    execution_context: dict[str, Any],
    messages: list[dict[str, Any]],
    layer: str,
    level: str,
) -> bool:
    """Fire at most one built-in nudge of *layer*, first matching row wins.

    Per-category caps and (for guidance) the enforcement/mode gate live inside each
    row's ``should_fire`` predicate, so this runner only walks ``_CORE_NUDGES`` in
    priority order and injects the first row whose predicate is satisfied.
    """
    for spec in _CORE_NUDGES:
        if spec.layer != layer:
            continue
        if not spec.should_fire(agent, query, active_mode, execution_context, level):
            continue
        content = spec.render(agent, execution_context)
        if not content or not str(content).strip():
            continue
        return _fire_nudge(
            execution_context, messages, spec.name, str(content), spec.budget_key,
        )
    return False

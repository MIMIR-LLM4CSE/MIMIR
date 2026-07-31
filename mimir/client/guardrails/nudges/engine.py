
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable

from ..workflow import (
    VALIDATION_RETRY_BUDGET,
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
    regression_nudge_message,
    state_nudge_message,
    unfinished_plan_nudge_message,
    todo_nudge_message,
    validation_nudge_message,
)
from ...context.execution_context import (
    backfill_execution_context,
    declared_edit_set_complete,
    has_discovery_evidence,
    idle_steps,
    known_existing_files,
    nudge_count,
)
from ...config.models import resolve_enforcement
from ...event_sink import emit
from ...config.constants import (
    CUSTOM_NUDGE_MAX_PER_QUERY,
    DISCOVERY_EVIDENCE_MIN_DISTINCT,
    NUDGE_MAX_BLAST_RADIUS,
    NUDGE_MAX_CREATION,
    NUDGE_MAX_DENIAL,
    NUDGE_MAX_DISCOVERY,
    NUDGE_MAX_DOC,
    NUDGE_MAX_ENV_CLEANUP,
    NUDGE_MAX_ENV_RESOLUTION,
    NUDGE_MAX_ERROR_RECOVERY,
    NUDGE_MAX_REGRESSION,
    NUDGE_MAX_STATE,
    NUDGE_MAX_TODO,
    NUDGE_MAX_UNFINISHED_PLAN,
    NUDGE_MAX_VALIDATION,
    NUDGE_STATE_IDLE_STEPS,
    NUDGE_VALIDATION_IDLE_STEPS,
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


# Guidance nudges permitted per (enforcement, mode). This is the single source for
# the enforcement dimension of the line; each nudge's own active_mode/situational
# gate still applies on top. Verification nudges and the plan-evidence gate are
# independent of this table. "off" is intentionally absent → empty set (and the
# caller short-circuits the whole guidance layer before it is consulted).
# Category strings match the labels passed to _fire_nudge (e.g. the documentation
# nudge's category is "doc", not "documentation").
_ALL_GUIDANCE = frozenset({
    "discovery", "env_resolution", "doc", "state",
    "blast_radius", "creation", "todo", "env_cleanup", "validation",
})
# Validation is agent-only: plan mode is read-only/discovery (no edits → no dirty
# files → nothing to validate), so the reminder can never legitimately fire there.
_ALL_GUIDANCE_PLAN = _ALL_GUIDANCE - {"validation"}
_GUIDANCE_BY_LEVEL_MODE: dict[tuple[str, str], frozenset[str]] = {
    # strict babysits everything (the branches still gate mode themselves, so this does
    # not actually leak agent-only nudges into plan mode) — minus validation in plan.
    ("strict", "agent"): _ALL_GUIDANCE,
    ("strict", "plan"): _ALL_GUIDANCE_PLAN,
    # light: the deliberate carve-out — only the nudges that guard a costly,
    # hard-to-detect, non-self-correcting mistake (blast_radius = breaking callers;
    # env_cleanup = leftover env side effects; validation = concluding on code that
    # was never checked — agent-only). Everything else is procedural hand-holding a
    # capable model does unprompted. Absent at "off" (guidance layer short-circuited).
    ("light", "agent"): frozenset({"blast_radius", "env_cleanup", "validation"}),
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
) -> bool:
    """Increment the per-category nudge counter, append the nudge, and signal it fired.

    Centralises the counter-increment + message-append + ``return True`` triplet that every
    nudge branch in ``maybe_append_nudge`` repeats. The per-nudge cap checks stay in the
    guarding ``if`` so each nudge keeps its own frequency limit.
    """
    counts = execution_context["nudge_counts"]
    counts[category] = counts.get(category, 0) + 1
    logger.debug("nudge fired: category=%s count=%d", category, counts[category])
    # Surface the injection to the front-end. A nudge is a message the *system* puts
    # in the user's turn slot; when it steers the model somewhere the user did not ask
    # for, the run is unreadable unless the injection itself is visible. Mirrors the
    # `steer_injected` event emitted for chat-while-busy steering.
    emit({"type": "nudge_injected", "category": category, "text": content})
    messages.append({"role": "user", "content": _NUDGE_TAG + content})
    return True


def _all_pending_budget_exhausted(execution_context: dict[str, Any]) -> bool:
    """Return True when every dirty file is either validated or budget-exhausted.

    In that case the model auto-escaped to conclude and cannot make further
    progress — suppress validation/state nudges to avoid confusing messages.
    """
    dirty = execution_context.get("dirty_written_files", set())
    if not dirty:
        return False
    validated = execution_context.get("validated_files", set())
    fail_counts = execution_context.get("validation_fail_count_by_file", {})
    return all(
        f in validated or int(fail_counts.get(f, 0)) >= VALIDATION_RETRY_BUDGET
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

    # If all dirty files are already validated (but workflow_state didn't reach
    # conclude via the normal path), treat the session as complete except for denials.
    dirty = execution_context.get("dirty_written_files", set())
    validated = execution_context.get("validated_files", set())
    if dirty and dirty.issubset(validated):
        return has_blocking_denials(execution_context)

    return (
        has_pending_validation(execution_context)
        or has_blocking_denials(execution_context)
        or (
            execution_context.get("code_mutation_started")
            and execution_context.get("workflow_state") != "conclude"
        )
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
    """Pending validation, budget left, editing paused, declared edit set complete.

    Enforcement-tiered (guidance layer): fires at ``strict`` and ``light`` but not at
    ``off`` — validating written code through bash is a reasoning shim a fully-trusted
    model is left to do on its own. The reminder is one of the light carve-outs because
    concluding on unvalidated code is a costly, hard-to-detect mistake.
    """
    return (
        _guidance_enabled("validation", enforcement=level, active_mode=active_mode)
        and has_pending_validation(execution_context)
        and nudge_count(execution_context, "validation") < NUDGE_MAX_VALIDATION
        and _retryable_pending_validation_exists(execution_context)
        and not _all_pending_budget_exhausted(execution_context)
        and idle_steps(execution_context) >= NUDGE_VALIDATION_IDLE_STEPS
        and declared_edit_set_complete(execution_context)
    )


def _validation_nudge_content(agent: Any, execution_context: dict[str, Any]) -> str:
    # Short pointer only — the full TIER 1 rationale already lives in the system
    # prompt; re-asserting it here as a user-role message would just duplicate the
    # strictest content through the more coercive channel. Pending validation is
    # handled uniformly regardless of whether the modified files are sources or tests.
    return validation_nudge_message(agent, execution_context)


def _should_nudge_denial(execution_context: dict[str, Any]) -> bool:
    """Blocking denials exist, budget left, and not already in the exhausted dead-end."""
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


def _should_nudge_regression(execution_context: dict[str, Any]) -> bool:
    """Budget left and an edited source has an on-disk test that was never run."""
    return (
        nudge_count(execution_context, "regression") < NUDGE_MAX_REGRESSION
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
    _CoreNudge(
        "regression", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_regression(ec),
        lambda agent, ec: regression_nudge_message(_untested_edited_sources(ec)),
    ),
    _CoreNudge(
        "unfinished_plan", "verification",
        lambda agent, query, mode, ec, level: _should_nudge_unfinished_plan(ec),
        lambda agent, ec: unfinished_plan_nudge_message(_required_unchecked_steps(ec)),
    ),
    # ── Guidance (reasoning shim; each predicate owns its enforcement/mode gate) ──
    # Validation leads the guidance layer: concluding on unvalidated code is the
    # costliest mistake, so it is a light carve-out (see _GUIDANCE_BY_LEVEL_MODE) and
    # tried before the softer reasoning nudges.
    _CoreNudge(
        "validation", "guidance",
        lambda agent, query, mode, ec, level: _should_nudge_validation(ec, level=level, active_mode=mode),
        lambda agent, ec: _validation_nudge_content(agent, ec),
    ),
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
        lambda agent, ec: state_nudge_message(ec),
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
        return _fire_nudge(execution_context, messages, spec.name, str(content))
    return False

"""The nudge cascade: at most one reminder per step, verification before guidance."""
from __future__ import annotations

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from .messages import (
    blast_radius_nudge_message,
    creation_nudge_message,
    denial_nudge_message,
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
    unwatched_run_nudge_message,
    validation_nudge_message,
)
from ...config.models import resolve_enforcement
from ...config.limits import (
    EXERCISE_BUDGET,
)
from ...core.extension_set import extensions_of
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from .exercise import exercise_hint, should_nudge_regression, should_nudge_unexercised_code
from .predicates import (
    first_failing_edit_path,
    has_declared_write_target,
    required_unchecked_steps,
    should_nudge_blast_radius,
    should_nudge_creation,
    should_nudge_denial,
    should_nudge_doc,
    should_nudge_env_cleanup,
    should_nudge_env_resolution,
    should_nudge_error_recovery,
    should_nudge_state,
    should_nudge_stuck_repair,
    should_nudge_todo,
    should_nudge_unfinished_plan,
    should_nudge_unwatched_run,
    should_nudge_validation,
    untested_edited_sources,
    worst_run_failure_streak,
)
from .reminders import inject_reminder


logger = logging.getLogger(__name__)


def _fire_nudge(
    execution_context: ExecutionContext,
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
    counts = execution_context.nudge_counts
    key = budget_key or category
    counts[key] = counts.get(key, 0) + 1
    logger.debug("nudge fired: category=%s budget=%s count=%d", category, key, counts[key])
    inject_reminder(messages, content, category=category,
                    execution_context=execution_context)
    return True


def maybe_append_nudge(
    *,
    agent: AgentPort,
    query: str,
    active_mode: str,
    execution_context: ExecutionContext,
    messages: list[dict[str, Any]],
) -> bool:
    """Append at most one workflow nudge, in priority order.

    The built-in table (``CORE_NUDGES``) comes first, then the rules of the agent's
    extension set; both answer the ``Nudge`` protocol. Two layers:

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
      ``GUIDANCE_BY_LEVEL_MODE`` (consulted inside each guidance predicate).

    Order per layer: core table first, then packs. Verification is tried before
    guidance, so it always wins.
    """
    level = resolve_enforcement(agent)
    for layer in ("verification", "guidance"):
        # Guidance nudges are a weak-model compatibility shim — skipped entirely when
        # the model's enforcement level is "off", core and application alike.
        if layer == "guidance" and level == "off":
            break
        for nudge in _cascade(agent, layer):
            if not nudge.fires(agent, query, active_mode, execution_context, level):
                continue
            content = nudge.text(agent, execution_context)
            if content.strip():
                return _fire_nudge(execution_context, messages, nudge.name, content,
                                   nudge.budget_key)
    _log_no_nudge(execution_context, level=level, active_mode=active_mode)
    return False


def nudge_pending(
    *,
    agent: AgentPort,
    query: str,
    active_mode: str,
    execution_context: ExecutionContext,
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
    level = resolve_enforcement(agent)
    layers = ("verification",) if level == "off" else ("verification", "guidance")
    return any(nudge.fires(agent, query, active_mode, execution_context, level)
               for layer in layers for nudge in _cascade(agent, layer))


def _log_no_nudge(
    execution_context: ExecutionContext,
    *,
    level: str,
    active_mode: str,
) -> None:
    """Record why no nudge was injected on a turn that asked for one.

    The counterpart to the fire-time log in :func:`_fire_nudge`: without it a suppression
    is invisible, and a false negative cannot be diagnosed after the fact. What is logged
    is the state the predicates actually read — a declared target, the workflow state,
    whether an edit happened, and the spent budgets — which is the trace worth having.
    Debug-level and lazily formatted, so it costs nothing when the logger is off.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug(
        "no nudge: level=%s mode=%s state=%s mutated=%s declared_target=%s counts=%s",
        level,
        active_mode,
        execution_context.workflow_state,
        bool(execution_context.code_mutation_started),
        has_declared_write_target(execution_context),
        execution_context.nudge_counts,
    )


class Nudge(Protocol):
    """What the cascade asks of a nudge, built-in (``CoreNudge``) or a pack's
    (``NudgeRule``)."""

    name: str        # nudge_counts category and toggle key
    layer: str       # "verification" | "guidance"

    @property
    def budget_key(self) -> str: ...

    def fires(self, agent: AgentPort, query: str, mode: str, ec: ExecutionContext,
              level: str) -> bool: ...

    def text(self, agent: AgentPort, ec: ExecutionContext) -> str: ...


def _enabled(agent: AgentPort, name: str) -> bool:
    return name not in (getattr(agent, "disabled_nudges", None) or set())


def _cascade(agent: AgentPort, layer: str) -> list[Nudge]:
    """The enabled nudges of *layer*, in the order they are tried: the core table, then
    the packs, so a built-in reality check always wins over an application's."""
    nudges: list[Nudge] = [n for n in CORE_NUDGES if n.layer == layer]
    nudges += extensions_of(agent).nudge_rules(layer)
    return [n for n in nudges if _enabled(agent, n.name)]


def _validation_nudge_content(agent: AgentPort, execution_context: ExecutionContext) -> str:
    # Short pointer only — the full TIER 1 rationale already lives in the system
    # prompt; re-asserting it here as a user-role message would just duplicate the
    # strictest content through the more coercive channel. Pending validation is
    # handled uniformly regardless of whether the modified files are sources or tests.
    return validation_nudge_message(agent, execution_context)


def maybe_inject_env_resolution(
    *,
    agent: AgentPort,
    active_mode: str,
    execution_context: ExecutionContext,
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
    if not _enabled(agent, "env_resolution"):
        return False
    level = resolve_enforcement(agent)
    if level == "off":
        return False
    if not should_nudge_env_resolution(
        execution_context, level=level, active_mode=active_mode
    ):
        return False
    return _fire_nudge(
        execution_context, messages, "env_resolution",
        env_resolution_nudge_message(execution_context),
    )


# The built-in nudges, as an ordered table of the same shape as a pack's ``NudgeRule``.
# Verification rows check reality and run at every enforcement level; guidance rows
# carry their enforcement/mode gate in their predicate (``tiers.guidance_enabled``).
# Order is priority: the first row of the layer that fires is the one sent. Each
# predicate also holds its per-category cap. A new built-in nudge is one row plus its
# predicate.
@dataclass(frozen=True)
class CoreNudge:
    name: str        # nudge_counts category / toggle key
    layer: str       # "verification" | "guidance"
    should_fire: Callable[[Any, str, str, dict[str, Any], str], bool]  # (agent, query, mode, ec, level)
    render: Callable[[Any, dict[str, Any]], str]                       # (agent, ec) -> text
    budget_key: str = ""  # counter charged when rows ration a shared budget; default = name

    # Per-category caps and (for guidance) the enforcement/mode gate live inside each
    # row's ``should_fire``, so the row answers the cascade's interface directly.

    def fires(self, agent: AgentPort, query: str, mode: str, ec: ExecutionContext, level: str) -> bool:
        return self.should_fire(agent, query, mode, ec, level)

    def text(self, agent: AgentPort, ec: ExecutionContext) -> str:
        return str(self.render(agent, ec) or "")


CORE_NUDGES: tuple[CoreNudge, ...] = (
    # ── Verification (reality checks; always run, independent of enforcement) ──
    CoreNudge(
        "denial", "verification",
        lambda agent, query, mode, ec, level: should_nudge_denial(ec),
        lambda agent, ec: denial_nudge_message(ec),
    ),
    # The client already attaches a structured failure diagnosis automatically, so
    # the model does not need to request one itself — just steer it to re-read.
    CoreNudge(
        "error_recovery", "verification",
        lambda agent, query, mode, ec, level: should_nudge_error_recovery(ec),
        lambda agent, ec: error_recovery_nudge_message(first_failing_edit_path(ec)),
    ),
    # Ahead of the three advisory rows below, because it is the only one of the four
    # that is required: a file the model modified and never checked is the one gap
    # ``needs_incomplete_finalization`` refuses to conclude over.
    # Before `validation`, and deliberately: while the model is going round the same
    # failure, telling it to finish checking is answering a question it is not stuck on.
    # Only one row speaks per turn, so this suppresses `validation` for as long as the
    # ladder is talking — which costs nothing, because pending validation still blocks
    # the conclusion through `needs_incomplete_finalization` either way.
    CoreNudge(
        "stuck_repair", "verification",
        lambda agent, query, mode, ec, level: should_nudge_stuck_repair(ec),
        lambda agent, ec: stuck_repair_nudge_message(worst_run_failure_streak(ec)),
    ),
    CoreNudge(
        "validation", "verification",
        lambda agent, query, mode, ec, level: should_nudge_validation(ec),
        lambda agent, ec: _validation_nudge_content(agent, ec),
    ),
    CoreNudge(
        "regression", "verification",
        lambda agent, query, mode, ec, level: should_nudge_regression(ec),
        lambda agent, ec: regression_nudge_message(untested_edited_sources(ec)),
        budget_key=EXERCISE_BUDGET,
    ),
    # Between them by design: `regression` covers an edit whose tests already exist,
    # this one the change that has no run at all behind it.
    CoreNudge(
        "unexercised", "verification",
        lambda agent, query, mode, ec, level: should_nudge_unexercised_code(agent, ec),
        lambda agent, ec: unexercised_code_nudge_message(
            sorted(ec.dirty_written_files or ()), exercise_hint(agent, ec)),
        budget_key=EXERCISE_BUDGET,
    ),
    CoreNudge(
        "unfinished_plan", "verification",
        lambda agent, query, mode, ec, level: should_nudge_unfinished_plan(ec),
        lambda agent, ec: unfinished_plan_nudge_message(required_unchecked_steps(ec)),
    ),
    # ── Guidance (reasoning shim; each predicate owns its enforcement/mode gate) ──
    CoreNudge(
        "env_resolution", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_env_resolution(ec, level=level, active_mode=mode),
        lambda agent, ec: env_resolution_nudge_message(ec),
    ),
    CoreNudge(
        "env_cleanup", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_env_cleanup(ec, level=level, active_mode=mode),
        lambda agent, ec: env_cleanup_nudge_message(ec),
    ),
    CoreNudge(
        "unwatched_run", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_unwatched_run(ec, level=level, active_mode=mode),
        lambda agent, ec: unwatched_run_nudge_message(ec),
    ),
    CoreNudge(
        "doc", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_doc(ec, level=level, active_mode=mode),
        lambda agent, ec: documentation_nudge_message(ec),
    ),
    CoreNudge(
        "state", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_state(ec, level=level, active_mode=mode),
        lambda agent, ec: state_nudge_message(agent, ec),
    ),
    CoreNudge(
        "blast_radius", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_blast_radius(ec, level=level, active_mode=mode),
        lambda agent, ec: blast_radius_nudge_message(ec),
    ),
    CoreNudge(
        "creation", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_creation(ec, level=level, active_mode=mode),
        lambda agent, ec: creation_nudge_message(ec),
    ),
    CoreNudge(
        "todo", "guidance",
        lambda agent, query, mode, ec, level: should_nudge_todo(ec, level=level, active_mode=mode),
        lambda agent, ec: todo_nudge_message(ec),
    ),
)

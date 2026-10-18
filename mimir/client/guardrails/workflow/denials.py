"""What refused approvals mean: the escalation ladder from reconsider to hand back."""
from __future__ import annotations

from ...config.limits import DENIAL_QUERY_HANDBACK_TOTAL, DENIAL_SCOPE_DROP_AFTER, DENIAL_SCOPE_HANDBACK_AFTER
from ...context.execution_context import ExecutionContext


# ── Denial escalation ladder ──────────────────────────────────────────────────
# A refusal means, in this order: 1. not this way (find another route), 2. that step is
# useless (drop it and say so), 3. stop (report and hand back). The model chooses; these
# stages take the earlier readings away as refusals of one approval scope accumulate.

STAGE_RECONSIDER: str = "reconsider"      # readings 1-3 all open


STAGE_DROP_OR_STOP: str = "drop_or_stop"  # reading 1 spent: drop the step or hand back


STAGE_HANDBACK: str = "handback"          # only reading 3 is left


def denial_scope_count(execution_context: ExecutionContext, scope: str) -> int:
    """How many times this approval scope has been refused this query."""
    if not scope:
        return 0
    return sum(
        1 for item in execution_context.denial_history
        if item.get("scope") == scope
    )


def denial_stage(execution_context: ExecutionContext, scope: str = "") -> str:
    """Which readings of a refusal are still open, for *scope* (or the query overall).

    A refusal already counted for *scope* must be in ``denial_history`` before this
    is called: the stage describes the situation the model is being handed, so the
    first refusal of a scope reports ``reconsider`` (all three readings open).
    """

    history = execution_context.denial_history
    total = len(history)
    scoped = denial_scope_count(execution_context, scope)

    # Stopping the run mid-prompt is the user saying "stop" in the plainest way there
    # is; it needs no accumulation.
    if any(item.get("kind") == "cancelled" for item in history):
        return STAGE_HANDBACK
    if scoped >= DENIAL_SCOPE_HANDBACK_AFTER or total >= DENIAL_QUERY_HANDBACK_TOTAL:
        return STAGE_HANDBACK
    if scoped >= DENIAL_SCOPE_DROP_AFTER or total >= DENIAL_QUERY_HANDBACK_TOTAL - 1:
        return STAGE_DROP_OR_STOP
    return STAGE_RECONSIDER


_STAGE_ORDER: tuple[str, ...] = (STAGE_RECONSIDER, STAGE_DROP_OR_STOP, STAGE_HANDBACK)


def worst_denial_stage(execution_context: ExecutionContext) -> str:
    """The furthest stage any refused scope has reached this query.

    What the layers that speak about the run as a whole (the nudge, the completion
    report) need: one scope at the end of the ladder governs the turn, even if another
    was only refused once.
    """
    history = execution_context.denial_history
    if not history:
        return STAGE_RECONSIDER
    scopes = {item.get("scope", "") for item in history}
    return max(
        (denial_stage(execution_context, scope) for scope in scopes),
        key=_STAGE_ORDER.index,
    )


def handback_required(execution_context: ExecutionContext) -> bool:
    """True once any refused scope has reached the end of the ladder."""
    return bool(execution_context.denial_history) and \
        worst_denial_stage(execution_context) == STAGE_HANDBACK


def approval_is_settled(execution_context: ExecutionContext, scope: str) -> bool:
    """True when a card for *scope* would re-ask a question already answered.

    From the second refusal of one scope, and for every scope past the hand-back.
    """
    return (
        denial_scope_count(execution_context, scope) >= DENIAL_SCOPE_DROP_AFTER
        or denial_stage(execution_context, scope) == STAGE_HANDBACK
    )


def handback_scopes(execution_context: ExecutionContext) -> list[dict]:
    """The refusal records whose scope reached ``handback``, newest first."""
    history = execution_context.denial_history
    seen: set = set()
    out: list[dict] = []
    for item in reversed(history):
        scope = item.get("scope", "")
        if scope in seen:
            continue
        seen.add(scope)
        if denial_stage(execution_context, scope) == STAGE_HANDBACK:
            out.append(item)
    return out

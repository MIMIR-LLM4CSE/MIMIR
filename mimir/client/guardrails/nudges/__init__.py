"""Soft guardrails: advisory nudges, the non-blocking sibling of ``policy``.

At most one reminder per step, tiered by enforcement level; policies hard-block. Both
read the ``execution_context`` blackboard and the ``guardrails.workflow`` state.
``plugins`` is the extension seam, as ``policy.plugins`` is for policies.
"""

from .cascade import maybe_append_nudge, maybe_inject_env_resolution, nudge_pending
from .exercise import exercise_route
from .plugins import (
    NudgeRule,
    VALID_NUDGE_LAYERS,
    register_nudge,
    rule_tier_enabled,
)
from .predicates import needs_incomplete_finalization
from .reminders import drop_transient_reminders, inject_reminder

__all__ = [
    "drop_transient_reminders",
    "exercise_route",
    "inject_reminder",
    "maybe_append_nudge",
    "maybe_inject_env_resolution",
    "needs_incomplete_finalization",
    "nudge_pending",
    "NudgeRule",
    "register_nudge",
    "rule_tier_enabled",
    "VALID_NUDGE_LAYERS",
]

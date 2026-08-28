"""Soft guardrails — advisory nudges (the non-blocking sibling of ``policy``).

Nudges append at most one advisory reminder per step, tiered by enforcement
level; policies hard-block. Both read the shared ``execution_context`` blackboard
and the ``guardrails.workflow`` state model. ``plugins`` mirrors
``policy.plugins`` (NudgeRule/NudgeRegistry/register_nudge ↔ PolicyCheck/...).
"""

from .engine import (
    inject_reminder,
    maybe_append_nudge,
    maybe_inject_env_resolution,
    needs_incomplete_finalization,
    nudge_pending,
)
from .plugins import (
    NudgeRule,
    NudgeRegistry,
    register_nudge,
    rule_tier_enabled,
    VALID_NUDGE_LAYERS,
)

__all__ = [
    "inject_reminder",
    "maybe_append_nudge",
    "maybe_inject_env_resolution",
    "needs_incomplete_finalization",
    "nudge_pending",
    "NudgeRule",
    "NudgeRegistry",
    "register_nudge",
    "rule_tier_enabled",
    "VALID_NUDGE_LAYERS",
]

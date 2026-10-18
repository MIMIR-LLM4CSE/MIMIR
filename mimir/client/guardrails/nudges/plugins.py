"""Application nudges registered on top of the core cascade, toggleable by name.

A ``NudgeRule`` runs after the core rows of its layer, one nudge per call at most.
Register at import time: ``register_nudge(NudgeRule(name, layer, predicate, render))``,
reading capabilities (``has_cap``), never tool names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from ...config.limits import CUSTOM_NUDGE_MAX_PER_QUERY
from ...core.extension_set import registration_target
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort

logger = logging.getLogger(__name__)

# (agent, query, active_mode, execution_context) -> should this nudge fire now?
NudgePredicate = Callable[[AgentPort, str, str, ExecutionContext], bool]
# (agent, execution_context) -> the message text (without the automated-reminder tag).
NudgeRender = Callable[[AgentPort, ExecutionContext], str]

VALID_NUDGE_LAYERS = ("verification", "guidance")


@dataclass(frozen=True)
class NudgeRule:
    name: str                 # toggle key + nudge_counts category
    layer: str                # "verification" (always on) | "guidance" (tier-gated)
    predicate: NudgePredicate
    render: NudgeRender
    priority: int = 100       # lower runs first
    # Guidance only: the (enforcement, mode) pairs at which this rule may fire.
    # None defaults to strict-in-both-modes — the safe, most-restrictive default, so a
    # pack that omits tiers does not leak guidance into 'light'/'off' deployments.
    tiers: frozenset[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.layer not in VALID_NUDGE_LAYERS:
            raise ValueError(
                f"NudgeRule.layer must be one of {VALID_NUDGE_LAYERS}, got {self.layer!r}"
            )

    # The ``Nudge`` interface the cascade asks of every nudge.

    @property
    def budget_key(self) -> str:
        return self.name

    def fires(self, agent: AgentPort, query: str, mode: str, ec: ExecutionContext, level: str) -> bool:
        """Whether the rule would fire now: under its per-query cap, in tier (guidance),
        and its predicate holds. A predicate that raises is "no", and is logged. The
        cascade skips a rule the user switched off before asking."""
        if (ec.nudge_counts or {}).get(self.name, 0) >= CUSTOM_NUDGE_MAX_PER_QUERY:
            return False
        if self.layer == "guidance" and not rule_tier_enabled(
                self, enforcement=level, active_mode=mode):
            return False
        try:
            return bool(self.predicate(agent, query, mode, ec))
        except Exception:
            logger.warning("nudge rule %r raised; skipped", self.name, exc_info=True)
            return False

    def text(self, agent: AgentPort, ec: ExecutionContext) -> str:
        try:
            return str(self.render(agent, ec) or "")
        except Exception:
            logger.warning("nudge rule %r raised; skipped", self.name, exc_info=True)
            return ""


# Default tier membership for a guidance rule that declares no explicit tiers: fire in
# 'strict' only, in both agent and plan modes. Mirrors the core "strict babysits
# everything" stance in tiers.GUIDANCE_BY_LEVEL_MODE.
_DEFAULT_TIERS: frozenset[tuple[str, str]] = frozenset({("strict", "agent"), ("strict", "plan")})


def rule_tier_enabled(rule: NudgeRule, *, enforcement: str, active_mode: str) -> bool:
    """True if a guidance *rule* may fire at this enforcement level and mode.

    Verification-layer rules ignore tiers (always eligible); the caller only consults
    this for guidance rules.
    """
    tiers = rule.tiers if rule.tiers is not None else _DEFAULT_TIERS
    return (enforcement, active_mode) in tiers


def register_nudge(rule: NudgeRule) -> None:
    """Add a nudge rule to the extension set being loaded (see ``core.extension_set``)."""
    registration_target().add_nudge(rule)

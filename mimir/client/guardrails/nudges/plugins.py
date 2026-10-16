"""Pluggable nudges — the extension seam for application-specific guidance.

The built-in nudge cascade in :mod:`guardrails.nudges.engine` (``maybe_append_nudge`` →
a verification layer that is always on + a guidance layer gated by
``_GUIDANCE_BY_LEVEL_MODE[(enforcement, mode)]``) is the fixed *core*. This module lets an
application register **additional** nudges on top — e.g. an "confirm this action is
authorized" reminder for a cybersecurity deployment — without editing core code.

Custom nudges run *after* their matching core layer and preserve the core invariants:
at most one nudge fires per ``maybe_append_nudge`` call, verification before guidance,
each injected with the ``_NUDGE_TAG`` prefix and ``role:"user"`` via the shared
``_fire_nudge`` helper. Unlike policies, nudges are **toggleable** (a disabled name in
``.mimir/preferences.json`` suppresses the rule).

A pack registers a :class:`NudgeRule` at import time::

    from mimir.client.extensions import NudgeRule, register_nudge
    from mimir.client.context.capabilities import names_with_cap, SENSITIVE

    register_nudge(NudgeRule(
        name="authz_reminder",
        layer="guidance",
        predicate=lambda agent, query, mode, ec: bool(names_with_cap(SENSITIVE, agent.tool_caps)),
        render=lambda agent, ec: "Before any sensitive action, confirm it is authorized.",
    ))

Reference *capabilities* (``names_with_cap``/``has_cap``), never literal tool names.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from ...config.limits import CUSTOM_NUDGE_MAX_PER_QUERY
from ...context.execution_context import ExecutionContext

logger = logging.getLogger(__name__)

# (agent, query, active_mode, execution_context) -> should this nudge fire now?
NudgePredicate = Callable[[Any, str, str, dict], bool]
# (agent, execution_context) -> the message text (without the automated-reminder tag).
NudgeRender = Callable[[Any, dict], str]

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

    # The interface every nudge of the cascade answers (see nudges.engine).

    @property
    def budget_key(self) -> str:
        return self.name

    def fires(self, agent: Any, query: str, mode: str, ec: ExecutionContext, level: str) -> bool:
        """Whether the rule would fire now: not switched off, under its per-query cap,
        in tier (guidance), and its predicate holds. A predicate that raises is "no": a
        bad pack cannot break the loop — but it is logged."""
        if self.name in (getattr(agent, "disabled_nudges", None) or set()):
            return False
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

    def text(self, agent: Any, ec: ExecutionContext) -> str:
        try:
            return str(self.render(agent, ec) or "")
        except Exception:
            logger.warning("nudge rule %r raised; skipped", self.name, exc_info=True)
            return ""


# Default tier membership for a guidance rule that declares no explicit tiers: fire in
# 'strict' only, in both agent and plan modes. Mirrors the core "strict babysits
# everything" stance in nudges.engine._GUIDANCE_BY_LEVEL_MODE.
_DEFAULT_TIERS: frozenset[tuple[str, str]] = frozenset({("strict", "agent"), ("strict", "plan")})


def rule_tier_enabled(rule: NudgeRule, *, enforcement: str, active_mode: str) -> bool:
    """True if a guidance *rule* may fire at this enforcement level and mode.

    Verification-layer rules ignore tiers (always eligible); the caller only consults
    this for guidance rules.
    """
    tiers = rule.tiers if rule.tiers is not None else _DEFAULT_TIERS
    return (enforcement, active_mode) in tiers


class _NudgeRegistry:
    """Process-global registry of application nudges (keyed by name)."""

    def __init__(self) -> None:
        self._rules: dict[str, NudgeRule] = {}

    def register(self, rule: NudgeRule) -> None:
        # Idempotent by name (see policy plugins for the rationale).
        self._rules[rule.name] = rule

    def rules(self, layer: str | None = None) -> list[NudgeRule]:
        """Registered rules (optionally filtered by layer), sorted by (priority, name)."""
        items = [r for r in self._rules.values() if layer is None or r.layer == layer]
        return sorted(items, key=lambda r: (r.priority, r.name))

    def names(self) -> list[str]:
        return sorted(self._rules)

    def clear(self) -> None:
        """Drop all registered rules (used by tests to isolate the global registry)."""
        self._rules.clear()


NudgeRegistry = _NudgeRegistry()


def register_nudge(rule: NudgeRule) -> None:
    """Public entry point a pack calls at import time to add a nudge."""
    NudgeRegistry.register(rule)

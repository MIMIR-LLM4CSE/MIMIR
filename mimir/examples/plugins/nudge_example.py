"""Example MIMIR nudge plugin — a toggleable authorization reminder.

Copy this into your plugins directory to activate it — `$MIMIR_PLUGINS_DIR` or
`<workspace>/.mimir/plugins/` — and drop the `.example` suffix. It is NOT auto-loaded
from `mimir/examples/`.

A *nudge* injects an advisory reminder into the conversation (it never blocks). This one
keys off the `SENSITIVE` capability and fires only when the connected toolset actually
exposes a sensitive tool. Nudges are **toggleable** (via `/nudges` + `.mimir/preferences.json`)
and tier-gated by enforcement level via their `tiers` field.

See PLUGINS_DETAILED.md for the full authoring guide.
"""
from __future__ import annotations

from typing import Any

from mimir.client.extensions import NudgeRule, register_nudge
from mimir.client.context.capabilities import SENSITIVE, names_with_cap


def _authz_predicate(agent: Any, query: str, active_mode: str, execution_context: dict) -> bool:
    """Fire only when the connected toolset exposes a sensitive capability."""
    return bool(names_with_cap(SENSITIVE, getattr(agent, "tool_caps", {})))


def _authz_render(agent: Any, execution_context: dict) -> str:
    return (
        "Before invoking any sensitive/side-effecting capability, confirm the action is "
        "authorized for this environment and least-privilege — prefer a read-only or "
        "dry-run capability first, and never transmit credentials in tool arguments."
    )


def register() -> None:
    """Register this plugin's descriptors (called for its import side effect)."""
    register_nudge(NudgeRule(
        name="authz_reminder",
        layer="guidance",
        predicate=_authz_predicate,
        render=_authz_render,
        # Fire in strict AND light (a security reminder is worth keeping even for a
        # capable model); omit "off" so a deployment that disables guidance stays quiet.
        tiers=frozenset({("strict", "agent"), ("strict", "plan"), ("light", "agent"), ("light", "plan")}),
    ))


register()

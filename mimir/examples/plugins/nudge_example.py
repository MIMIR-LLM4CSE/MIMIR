"""Example nudge plugin: a toggleable reminder when a sensitive tool is connected.

Copy into ``.mimir/plugins/`` (or ``$MIMIR_PLUGINS_DIR``) without the ``.example`` suffix.
See PLUGINS_DETAILED.md.
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

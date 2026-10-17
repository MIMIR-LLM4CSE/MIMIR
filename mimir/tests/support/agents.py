"""Stand-in agents: the parts every layer reads off an ``AgentPort``, plus whatever a
test adds or overrides."""
from __future__ import annotations

import types
from typing import Any

from mimir.client.config.toggles import Toggles
from mimir.client.context.carry import CarryContext
from mimir.client.core.extension_set import AMBIENT, ExtensionSet
from mimir.client.core.turn_state import TurnState


def agent_parts() -> dict[str, Any]:
    """Fresh copies of what a real agent is built with: agent mode, no tools yet,
    nothing switched off, the ambient extensions, an idle turn, an empty carry."""
    return {
        "mode": "agent",
        "tool_caps": {},
        "extensions": ExtensionSet(parent=AMBIENT),
        "toggles": Toggles({}, persist=False),
        "turn": TurnState(),
        "carry": CarryContext(),
    }


def fake_agent(**fields: Any) -> types.SimpleNamespace:
    """An agent made of :func:`agent_parts` and *fields*, which win."""
    return types.SimpleNamespace(**{**agent_parts(), **fields})


class FakeAgent:
    """Base for a test's agent class: each of :func:`agent_parts` the class does not
    define itself, set on the instance."""

    def __init__(self) -> None:
        for name, value in agent_parts().items():
            if not hasattr(type(self), name):
                setattr(self, name, value)

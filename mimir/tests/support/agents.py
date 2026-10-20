"""Stand-in agents: the parts every layer reads off an ``AgentPort``, plus whatever a
test adds or overrides."""
from __future__ import annotations

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
        "model": "",
        "tool_caps": {},
        "tool_owner": {},
        "tools": [],
        "skills": {},
        "resources": {},
        "sessions": {},
        "server_env": {},
        "extensions": ExtensionSet(parent=AMBIENT),
        "toggles": Toggles({}, persist=False),
        "turn": TurnState(),
        "carry": CarryContext(),
        "workspace_root": "",
        "session_id": None,
        # Unset settings: what the engine reads as "not chosen".
        "thinking": None,
        "thinking_depth": None,
        "thinking_budget": -1,
        "temperature": None,
        "context_mode": "full",
        "enforcement": None,
        "subagent_level": "explore",
        "backend": "",
        "max_context_tokens": 0,
        "max_answer_tokens": 0,
        "non_interactive": False,
        # No front-end lending anything.
        "poll_steer": None,
        "register_background_job": None,
        "watched_background_jobs": None,
        # No model behind it to summarise with: the loop does not compact.
        "compact_messages": None,
    }


def fake_agent(**fields: Any) -> "FakeAgent":
    """An agent made of :func:`agent_parts` and *fields*, which win."""
    agent = FakeAgent()
    for name, value in fields.items():
        setattr(agent, name, value)
    return agent


class FakeAgent:
    """Base for a test's agent class: each of :func:`agent_parts` the class does not
    define itself, set on the instance, and the port's derived views of them."""

    def __init__(self) -> None:
        for name, value in agent_parts().items():
            if not hasattr(type(self), name):
                setattr(self, name, value)

    def request_user_question(self, questions: list, origin: dict | None = None,
                              timeout_secs: float | None = None) -> dict:
        return {}   # nobody to ask: no answer

    def advertised_tools(self) -> list[dict]:
        return list(self.tools)

    def advertised_tools_for_mode(self, active_mode: str | None = None) -> list[dict]:
        from mimir.client.config.models import READONLY_MODES
        from mimir.client.guardrails.policy.toollist import tools_for_readonly_mode
        tools = self.advertised_tools()
        mode = active_mode or self.mode
        if mode in READONLY_MODES:
            tools = tools_for_readonly_mode(tools, self.tool_caps, mode=mode)
        return tools

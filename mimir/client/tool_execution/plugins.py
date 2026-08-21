"""Pluggable post-tool hooks — the extension seam for user-owned verification.

The policy seam (:mod:`guardrails.policy.plugins`) runs *before* a tool call and can
block it; the nudge seam (:mod:`guardrails.nudges.plugins`) runs at the *end of a turn*
and can re-prompt. Between them sits the one thing neither can do: react to what a tool
call actually produced. That is what this seam is for — a check the machine performs and
records, next to the verdict the model states about itself.

A hook runs after a successful tool call, outside the call's own timeout budget, and
returns text appended to the tool result (empty string for "nothing to say"). It cannot
block: the call already happened. What it *can* do is write to the blackboard through
the ordinary entry points, and the turn-end gates read that — a hook that runs the
project's tests and records a failure makes the existing verification nudges refuse the
conclusion, with no new blocking mechanism.

A pack registers a :class:`PostToolRule` at import time::

    from mimir.client.extensions import PostToolRule, register_post_tool
    from mimir.client.context.capabilities import EDIT, has_cap

    async def _project_checks(agent, tool_name, arguments, result, execution_context):
        if not has_cap(tool_name, EDIT, agent.tool_caps):
            return ""
        out = await agent._run_tool(
            "bash_run", {"command": "pytest -q"}, execution_context=execution_context,
        )
        return "" if '"status": "ok"' in out else "\\n\\nPROJECT_CHECK: the suite is red."

    register_post_tool(PostToolRule(name="project_checks", run=_project_checks))

``run`` may be sync or async. Reference *capabilities*, never literal tool names.

Running a command goes through the ordinary tool path, so the approval gate still
applies: a hook cannot execute shell the user never agreed to. An "always" grant on the
command prefix covers it for the session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# (agent, tool_name, arguments, result, execution_context) -> annotation appended to the
# tool result, or "" / None for nothing. May be a coroutine function.
PostToolFn = Callable[[Any, str, dict, str, "dict | None"], "Any"]


@dataclass(frozen=True)
class PostToolRule:
    name: str
    run: PostToolFn
    order: int = 100     # lower runs first


class _PostToolRegistry:
    """Process-global registry of application post-tool hooks (keyed by name)."""

    def __init__(self) -> None:
        self._rules: dict[str, PostToolRule] = {}

    def register(self, rule: PostToolRule) -> None:
        # Idempotent by name (see policy plugins for the rationale).
        self._rules[rule.name] = rule

    def rules(self) -> list[PostToolRule]:
        return sorted(self._rules.values(), key=lambda r: (r.order, r.name))

    def names(self) -> list[str]:
        return sorted(self._rules)

    def clear(self) -> None:
        """Drop all registered hooks (used by tests to isolate the global registry)."""
        self._rules.clear()


PostToolRegistry = _PostToolRegistry()


def register_post_tool(rule: PostToolRule) -> None:
    """Public entry point a pack calls at import time to add a post-tool hook."""
    PostToolRegistry.register(rule)

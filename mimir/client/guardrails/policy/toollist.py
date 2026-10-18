"""Which tools the model sees each query.

Only two things withhold a tool: the user (a server switched off, in
``agent_core.advertised_tools``) and the mode (read-only modes, planning tools), both
from declared capabilities. Nothing is pruned by the query's wording.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config.settings import DEFAULT_SUBAGENT_LEVEL, subagent_level_allows
from ...config.models import READONLY_MODES
from ...tools.capabilities import (
    PLAN_BLOCKED,
    PLAN_READONLY,
    TASK_PLANNING,
    has_cap,
    names_with_arg_role,
    names_with_cap,
    reserved_for_main,
)
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort

logger = logging.getLogger(__name__)


def blocked_tools_for_context(
    query: str, execution_context: ExecutionContext | None, tool_caps: Any = None,
) -> set[str]:

    if execution_context is None:
        return set()

    # Nothing is blocked here: a tool hidden for the whole query on a guess about the
    # request has no way back. Rewriting an unread file is refused at call time.
    return set()


def tools_for_context(
    *,
    query: str,
    execution_context: ExecutionContext | None,
    tools: list[dict[str, Any]],
    tool_caps: Any = None,
) -> list[dict[str, Any]]:
    """The advertised tools minus :func:`blocked_tools_for_context` (empty today).

    The seam for a query-stable block; state-dependent checks belong at call time.
    """
    blocked = blocked_tools_for_context(query, execution_context, tool_caps)
    if not blocked:
        return tools
    return [
        tool for tool in tools
        if (tool.get("function") or {}).get("name", "") not in blocked
    ]


def hidden_planning_tools(mode: str, tool_caps: Any = None, *, exploring: bool = False) -> set[str]:
    """Plan-writing tools *mode* does not expose, by capability.

    Ask hides every ``TASK_PLANNING`` writer; plan hides the checklist (``plan_steps``), and
    while *exploring* the plan document too. A tool not shown needs no prohibition.
    """
    if mode == "ask":
        return names_with_cap(TASK_PLANNING, tool_caps)
    if mode == "plan":
        hidden = names_with_arg_role("plan_steps", tool_caps)
        if exploring:
            hidden = hidden | names_with_arg_role("plan_title", tool_caps)
        return hidden
    return set()


def tools_for_readonly_mode(
    tools: list[dict[str, Any]], tool_caps: Any = None, mode: str = "",
    *, exploring: bool = False,
) -> list[dict[str, Any]]:
    """The tools a read-only mode keeps: no ``PLAN_BLOCKED`` tool, no unneeded planning tool.

    The dual-use bash tool stays; its exec use is refused at call time
    (``readonly_guard.filter_readonly_tool_calls``).
    """
    hidden = names_with_cap(PLAN_BLOCKED, tool_caps) | hidden_planning_tools(
        mode, tool_caps, exploring=exploring,
    )
    return [
        tool for tool in tools
        if tool.get("function", {}).get("name") not in hidden
    ]


def grantable_tools(agent: AgentPort) -> dict[str, str]:
    """What the orchestrator may hand a sub-agent now: ``{tool: server}``.

    Filtered by the user's toggles, the orchestrator's mode, the sub-agent rung (writing
    tools only at "build", always in a worktree) and :func:`reserved_for_main`. The server
    travels with each name: the child connects servers, not tools.
    """
    hidden = reserved_for_main(agent.tool_caps)
    tools = agent.advertised_tools()
    level = getattr(agent, "subagent_level", DEFAULT_SUBAGENT_LEVEL)
    mode = agent.mode
    if mode in READONLY_MODES or not subagent_level_allows(level, "build"):
        tools = tools_for_readonly_mode(
            tools, agent.tool_caps, mode=mode if mode in READONLY_MODES else "ask")
    granted: dict[str, str] = {}
    for tool in tools:
        name = tool.get("function", {}).get("name", "")
        owner = agent.tool_owner.get(name)
        if name and owner and name not in hidden:
            granted[name] = owner
    return granted


def grantable_writers(agent: AgentPort, granted: dict[str, str] | None = None) -> list[str]:
    """Which granted tools would change the tree a sub-agent works in."""
    granted = grantable_tools(agent) if granted is None else granted
    return sorted(
        name for name in granted
        if has_cap(name, PLAN_BLOCKED, agent.tool_caps)
        or has_cap(name, PLAN_READONLY, agent.tool_caps)
    )


def tools_for_plan_mode(
    tools: list[dict[str, Any]], tool_caps: Any = None, *, exploring: bool = False,
) -> list[dict[str, Any]]:
    """The read-only subset for plan mode. Its own name because plan_loop imports it.

    ``exploring`` withholds the plan-document tool for the explore phase; plan_loop
    rebuilds the list once when the evidence bar is met.
    """
    return tools_for_readonly_mode(tools, tool_caps, mode="plan", exploring=exploring)

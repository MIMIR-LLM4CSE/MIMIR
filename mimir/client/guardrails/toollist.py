"""Per-query tool-list construction (which tools the model sees each query).

Distinct from the call-time precondition gates: this module decides the *visible*
tool set. Exactly two things withhold a tool now, and neither is a guess about the
task:

* **The user**, by switching a server off — applied upstream in
  ``agent_core.advertised_tools`` (a soft-hide: the subprocess stays connected). The
  user has judged those tools irrelevant to what they are doing, and that judgement
  is theirs to make; nothing here second-guesses it.
* **The mode**, here: a read-only mode does not get the write/exec surface, and a
  mode does not get the plan-writing tools it has no use for. Both are read from
  capabilities the servers declare, so the decision is a fact.

Everything else is sent, to every model. Nothing is pruned by the query's wording: a
wrong guess about the task removes a capability, and a list that changes per request
misses the prompt cache every time. The inputs are the mode and the user's server
toggles, which do not move during a query unless the user moves them.

This is prompt-construction (which tools the model sees *before* it chooses), the
counterpart to ``tool_execution`` which runs the call the model chose *after*. It is
not a precondition gate, which is why it is not in ``policy``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config.settings import DEFAULT_SUBAGENT_LEVEL, subagent_level_allows
from ..config.models import READONLY_MODES
from ..context.capabilities import (
    PLAN_BLOCKED, PLAN_READONLY, TASK_PLANNING, has_cap, names_with_arg_role,
    names_with_cap, reserved_for_main,
)
from ..context.execution_context import ExecutionContext
from ..core.ports import AgentPort

logger = logging.getLogger(__name__)


def blocked_tools_for_context(
    query: str, execution_context: ExecutionContext | None, tool_caps: Any = None,
) -> set[str]:

    if execution_context is None:
        return set()

    # Nothing is blocked here, on purpose. Intent is not guessed from the request's
    # vocabulary: a refactor that splits a module is pure file creation, and a tool
    # hidden for the whole query has no way back. Rewriting a file never read is refused
    # at call time, by fact (policy.write.check_write_policy).
    #
    # Kept as a function, and still called, so the seam is here if a genuinely
    # query-stable block is ever needed. Anything state-dependent belongs in
    # evaluate_tool_preconditions instead.
    return set()


def tools_for_context(
    *,
    query: str,
    execution_context: ExecutionContext | None,
    tools: list[dict[str, Any]],
    tool_caps: Any = None,
) -> list[dict[str, Any]]:
    """The advertised tools, minus whatever :func:`blocked_tools_for_context` withholds.

    That set is empty today, so in practice this returns the list unchanged — which is
    the point. The mode filter the caller applies first is the only thing that removes a
    tool; nothing here reasons about what the task is *about*.

    Kept as a function, and still called, because the seam is where a genuinely
    query-stable block would belong if one is ever needed. Anything state-dependent
    belongs in ``evaluate_tool_preconditions`` instead, at call time.
    """
    blocked = blocked_tools_for_context(query, execution_context, tool_caps)
    if not blocked:
        return tools
    return [
        tool for tool in tools
        if (tool.get("function") or {}).get("name", "") not in blocked
    ]


def hidden_planning_tools(mode: str, tool_caps: Any = None, *, exploring: bool = False) -> set[str]:
    """Plan-writing tools *mode* does not expose, by capability rather than by name.

    ASK answers and records nothing, so every ``TASK_PLANNING`` writer is hidden.
    PLAN records the prose document only — the ordered checklist is written after the
    user approves, at the start of the execution — so the tool declaring the
    ``plan_steps`` arg-role is hidden. Withdrawing the tool is what lets the prompt
    drop the matching "do not write a plan / a checklist" prohibitions: a tool the
    model cannot see needs no rule, no nudge, and no prompt tokens.

    ``exploring`` extends that to the document tool itself (the ``plan_title``
    arg-role) for plan mode's first phase. A plan mode that offers the document from
    the first turn, under a nudge calling the plan mandatory, makes a plan *to
    explore* the cheapest way out — so the exploration is not asked for in prose, it
    is the only thing the tool list allows.
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
    """Return the exploration-safe subset of tools allowed in a read-only mode.

    Shared by every mode in ``READONLY_MODES`` (plan, ask). Write, execution, and
    mutation tools (PLAN_BLOCKED) are stripped, along with the plan-writing tools
    *mode* has no use for (see :func:`hidden_planning_tools`). Everything else —
    search, read, inspect, platform query, memory read, todo read, and the dual-use
    PLAN_READONLY exec tool (kept visible so the model can run read-only discovery
    commands; its exec use is rejected at call time by
    :func:`readonly_guard.filter_readonly_tool_calls`) — is kept so the model can
    gather the evidence its answer or plan is grounded in.
    """
    hidden = names_with_cap(PLAN_BLOCKED, tool_caps) | hidden_planning_tools(
        mode, tool_caps, exploring=exploring,
    )
    return [
        tool for tool in tools
        if tool.get("function", {}).get("name") not in hidden
    ]


def grantable_tools(agent: AgentPort) -> dict[str, str]:
    """What the orchestrator may hand to a sub-agent right now: ``{tool: server}``.

    The same three filters that decide what *it* can use, in the same order, plus two:

    * the user's server toggles, already applied by ``agent.advertised_tools()``;
    * its own mode — a read-only mode has no write surface to give away, which is why
      a sub-agent cannot be the way around plan mode;
    * the **sub-agent rung** the user set (``config.settings.SUBAGENT_LEVELS``): at
      "explore" the surface is read-only whatever mode the orchestrator is in, because
      a child that writes is something the user hands over, not something the model
      decides. At "build" the writing tools may be granted, and the child that gets
      one works in a copy of the repository — never in this tree;
    * :func:`~context.capabilities.reserved_for_main`, the tools a child is never
      granted, read off the capabilities the servers declare.

    The owning server travels with each name because the child connects servers, not
    tools: it starts only the ones its granted tools live in, instead of all of them.
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
    """Which granted tools would have a sub-agent change the tree it works in.

    Read off the capabilities, here, because the spawning server sees only names: it
    connects the servers a grant needs and has no registry until it does. What it does
    with the answer is arbitration — two children editing the same files overwrite each
    other in silence, so in a shared tree they take turns.
    """
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

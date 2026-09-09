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

Everything else is sent, to every model. Two filters used to trim the list further —
pruning whole tool families the query's wording did not mention, and capping the
total by relevance — and both were removed deliberately:

* They guessed the task's subject from its opening sentence, and a guess that is
  wrong *removes a capability*. The recorded failure is an auto-optimisation task
  whose request said "optimise this solver": the plan mode that preceded it did not
  prune, so the plan was built around the surrogate tools, and the execution that
  followed rebuilt the list from the request alone, dropped them, and fell back to
  the shell. Every replacement signal measured out fragile in the same way.
* They cost the very thing they looked like they were saving. The full list is
  *constant*, so it sits in the prompt prefix and is paid for once per session; a
  list that depends on the request changes with every request and misses that cache
  each time.

What remains is query-stable by construction: the mode and the user's server toggles
are the only inputs, and neither moves during a query unless the user moves it.

This is prompt-construction (which tools the model sees *before* it chooses), the
counterpart to ``tool_execution`` which runs the call the model chose *after*. It
moved here out of ``policy/engine.py`` because it is not a precondition gate.
"""

from __future__ import annotations

import logging
from typing import Any

from ..context.capabilities import (
    PLAN_BLOCKED, TASK_PLANNING, names_with_arg_role, names_with_cap,
)
from ..context.execution_context import bootstrap_engine_context as _bootstrap_engine_context

logger = logging.getLogger(__name__)


def blocked_tools_for_context(
    query: str, execution_context: dict[str, Any] | None, tool_caps: Any = None,
) -> set[str]:
    execution_context = _bootstrap_engine_context(execution_context)

    if execution_context is None:
        return set()

    # Nothing is blocked here any more, and the empty set is deliberate — see below.
    #
    # This used to drop every CONTENT_WRITE tool (`write_file`, `append_file`) whenever
    # the query read as edit-flavoured, on the reasoning that "appending is effectively
    # creating new content, not a surgical edit". Two things were wrong with it.
    #
    # It guessed intent from vocabulary, and the guess is not sound: "refactor" is an
    # edit word, but the most ordinary refactor there is — splitting a module into a
    # package — is pure file creation. The query cannot answer "will this need a new
    # file?"; only the work can.
    #
    # And it enforced the guess by AMPUTATION. Hiding here is query-stable and therefore
    # irreversible for the whole query, with no call-time path back.
    # A model that had planned a package split was left with
    # `replace_in_file` alone — which cannot create a file — and had no way to say so:
    # it created the directory, read what it needed, and then returned an EMPTY TURN at
    # the exact step where the write belonged, twice, in two recorded sessions on the
    # same request. The visible symptom was an answer ending "Creating the package files
    # now:" with nothing created.
    #
    # The concern behind it was real but already met, precisely and at call time:
    # `policy.write.check_write_policy` refuses an OVERWRITE-capable tool aimed at a file
    # that is known to exist and was never read. That gate tells rewriting a file from
    # creating one by FACT rather than by keyword, which is the distinction that was
    # wanted. `query_prefers_existing_file_edits` keeps its place in the nudge layer,
    # where "prefer a surgical edit" is advice the model can weigh, rather than a
    # capability taken away from it.
    #
    # Kept as a function, and still called, so the seam is here if a genuinely
    # query-stable block is ever needed. Anything state-dependent belongs in
    # evaluate_tool_preconditions instead.
    return set()


def tools_for_context(
    *,
    query: str,
    execution_context: dict[str, Any] | None,
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


def tools_for_plan_mode(
    tools: list[dict[str, Any]], tool_caps: Any = None, *, exploring: bool = False,
) -> list[dict[str, Any]]:
    """The read-only subset for plan mode. Its own name because plan_loop imports it.

    ``exploring`` withholds the plan-document tool for the explore phase; plan_loop
    rebuilds the list once when the evidence bar is met.
    """
    return tools_for_readonly_mode(tools, tool_caps, mode="plan", exploring=exploring)

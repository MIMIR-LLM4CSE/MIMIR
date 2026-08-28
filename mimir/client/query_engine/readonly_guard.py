"""Call-time guard for the read-only modes (plan, ask).

``tools_for_readonly_mode`` already hides the PLAN_BLOCKED tools and the plan-writing
tools the mode has no use for, but some models still hallucinate calls to tools that
were never advertised, and the dual-use PLAN_READONLY exec tool stays visible on
purpose (discovery commands). This module is the defence in depth: it drops those
classes of call and feeds the model a tool-role error explaining why, so it can
correct course.

Extracted from ``plan_loop.py`` when ask mode started sharing the same guard.
"""
from __future__ import annotations

import json
from typing import Any

from ..event_sink import emit
from ..context.capabilities import (
    PLAN_BLOCKED, PLAN_READONLY, has_cap, readonly_invocation_spec,
)
from ..guardrails.policy.bash_classify import bash_command_is_readonly
from .streaming import _to_dict
from .toollist import hidden_planning_tools

# Why a mode has no plan-writing tool, told to a model that called one anyway.
_NO_PLANNING_REASON = {
    "plan": (
        "Record the written plan document only — the ordered checklist is written once "
        "the user has approved the plan, right before the work starts."
    ),
    "ask": (
        "This is a question-answering turn: nothing is planned and nothing is recorded. "
        "Answer in prose."
    ),
}


def filter_readonly_tool_calls(
    tool_calls: list[Any],
    *,
    agent: Any,
    messages: list[dict],
    mode_label: str,
) -> list[Any]:
    """Return the subset of *tool_calls* that may run in a read-only mode.

    A PLAN_BLOCKED call (write / execution / mutation) is always dropped, as is a
    plan-writing call the mode does not expose (see
    :func:`toollist.hidden_planning_tools`). A PLAN_READONLY call is dual-use: it is
    kept when its command is read-only (search, list, read) and dropped when it would
    run or mutate anything. Every rejection appends a ``role="tool"`` error to
    *messages* and emits a status line. ``mode_label`` ("plan" / "ask") selects the
    hidden planning set and shapes those texts.
    """
    no_planning = hidden_planning_tools(mode_label, agent.tool_caps)
    safe: list[Any] = []
    for tc in tool_calls:
        fn = _to_dict(_to_dict(tc).get("function", {}))
        tc_name = fn.get("name", "")
        if has_cap(tc_name, PLAN_BLOCKED, agent.tool_caps):
            emit({"type": "status", "text": f"  ⚠ Blocked '{tc_name}' in {mode_label} mode"})
            messages.append({"role": "tool", "content": json.dumps({"status": "error", "error": f"Tool '{tc_name}' is not available in {mode_label} mode. Use exploration tools only."})})
        elif tc_name in no_planning:
            emit({"type": "status", "text": f"  ⚠ Blocked '{tc_name}' in {mode_label} mode"})
            messages.append({"role": "tool", "content": json.dumps({"status": "error", "error": f"Tool '{tc_name}' is not available in {mode_label} mode. " + _NO_PLANNING_REASON.get(mode_label, "")})})
        elif has_cap(tc_name, PLAN_READONLY, agent.tool_caps):
            args = agent._normalize_arguments(fn.get("arguments") or {})
            # Two shapes of dual-use tool. One declares which argument value makes the
            # call read-only; the other carries a shell command, which only the
            # classifier can judge.
            spec = readonly_invocation_spec(tc_name, agent.tool_caps)
            if spec is not None:
                value = str(args.get(spec["arg"], "")).strip().lower()
                allowed = [v.lower() for v in spec["values"]]
                # An omitted argument is the tool's own default, not a violation: a
                # tool declaring this spec makes its read-only invocation the default.
                if not value or value in allowed:
                    safe.append(tc)
                else:
                    emit({"type": "status", "text": f"  ⚠ Blocked '{tc_name}' ({spec['arg']}={value or 'unset'}) in {mode_label} mode"})
                    messages.append({"role": "tool", "content": json.dumps({"status": "error", "error": f"In {mode_label} mode '{tc_name}' is available only with {spec['arg']}={' or '.join(allowed)} — nothing may write, execute or mutate. Re-issue the call that way."})})
                continue
            cmd = args.get("command", "")
            if bash_command_is_readonly(cmd):
                safe.append(tc)
            else:
                emit({"type": "status", "text": f"  ⚠ Blocked exec command in {mode_label} mode: {tc_name}"})
                messages.append({"role": "tool", "content": json.dumps({"status": "error", "error": f"Only read-only discovery commands are allowed in {mode_label} mode. '{cmd}' runs code or mutates state — use it for inspection only (e.g. search, list, read files)."})})
        else:
            safe.append(tc)
    return safe

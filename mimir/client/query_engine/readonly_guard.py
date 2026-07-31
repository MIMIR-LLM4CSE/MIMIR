"""Call-time guard for the read-only modes (plan, ask).

``tools_for_readonly_mode`` already hides PLAN_BLOCKED tools from the model, but
some models still hallucinate calls to tools that were never advertised, and the
dual-use PLAN_READONLY exec tool stays visible on purpose (discovery commands).
This module is the defence in depth: it drops both classes of unsafe call and
feeds the model a tool-role error explaining why, so it can correct course.

Extracted from ``plan_loop.py`` when ask mode started sharing the same guard.
"""
from __future__ import annotations

import json
from typing import Any

from ..event_sink import emit
from ..context.capabilities import PLAN_BLOCKED, PLAN_READONLY, has_cap
from ..guardrails.policy.bash_classify import bash_command_is_readonly
from .streaming import _to_dict


def filter_readonly_tool_calls(
    tool_calls: list[Any],
    *,
    agent: Any,
    messages: list[dict],
    mode_label: str,
) -> list[Any]:
    """Return the subset of *tool_calls* that may run in a read-only mode.

    A PLAN_BLOCKED call (write / execution / mutation) is always dropped. A
    PLAN_READONLY call is dual-use: it is kept when its command is read-only
    (search, list, read) and dropped when it would run or mutate anything.
    Every rejection appends a ``role="tool"`` error to *messages* and emits a
    status line. ``mode_label`` ("plan" / "ask") only shapes those texts.
    """
    safe: list[Any] = []
    for tc in tool_calls:
        fn = _to_dict(_to_dict(tc).get("function", {}))
        tc_name = fn.get("name", "")
        if has_cap(tc_name, PLAN_BLOCKED, agent.tool_caps):
            emit({"type": "status", "text": f"  ⚠ Blocked '{tc_name}' in {mode_label} mode"})
            messages.append({"role": "tool", "content": json.dumps({"status": "error", "error": f"Tool '{tc_name}' is not available in {mode_label} mode. Use exploration tools only."})})
        elif has_cap(tc_name, PLAN_READONLY, agent.tool_caps):
            cmd = agent._normalize_arguments(fn.get("arguments") or {}).get("command", "")
            if bash_command_is_readonly(cmd):
                safe.append(tc)
            else:
                emit({"type": "status", "text": f"  ⚠ Blocked exec command in {mode_label} mode: {tc_name}"})
                messages.append({"role": "tool", "content": json.dumps({"status": "error", "error": f"Only read-only discovery commands are allowed in {mode_label} mode. '{cmd}' runs code or mutates state — use it for inspection only (e.g. search, list, read files)."})})
        else:
            safe.append(tc)
    return safe

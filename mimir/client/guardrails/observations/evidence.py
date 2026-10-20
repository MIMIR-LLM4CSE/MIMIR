"""Statements about evidence: verdicts on runs, and denials a later call clears."""
from __future__ import annotations

from typing import Any

from ...protocol.event_sink import emit
from ..verification.verdict import apply_verdict
from ...tools.capabilities import JUDGE, arg_role, has_cap
from ...context.execution_context import ExecutionContext
from .call import Call


def record_denial(execution_context: ExecutionContext, *, tool: str, path: str,
                  fallback_tools: list[str], scope: Any, kind: str, reason: str) -> None:
    """Record a refusal in the open set (``denied_tool_calls``) and the scope history.

    The open set clears when the action later succeeds (:func:`denial_clearing`); the
    history does not, since the escalation ladder counts it.
    """
    execution_context.denials.denied_tool_calls.append({
        "tool": tool, "path": path, "fallback_tools": fallback_tools,
        "scope": scope, "kind": kind, "reason": reason,
    })
    execution_context.denials.denial_history.append({
        "tool": tool, "scope": scope, "kind": kind, "reason": reason,
    })


def denial_clearing(call: Call) -> None:
    """Clear a recorded denial once the same tool+path later succeeds."""
    execution_context = call.execution_context
    if call.status != "ok":
        return
    resolved_path = call.agent.normalize_workspace_path(
        call.arguments.get("path")
        or call.arguments.get("filepath")
        or call.arguments.get("subdir")
        or call.arguments.get("directory")
    )
    denied_calls = execution_context.denials.denied_tool_calls
    execution_context.denials.denied_tool_calls = [
        item for item in denied_calls
        if not (
            item.get("tool") == call.tool_name
            and item.get("path") == resolved_path
        )
    ]
    if not execution_context.denials.denied_tool_calls:
        execution_context.nudges.nudge_counts["denial"] = 0


def verdict_tool(call: Call) -> None:
    """The model judged a run's output: settle the runs it speaks for.

    Found by capability and read through its arg-roles. Settled runs are announced per
    call id; a statement that settled nothing announces nothing.
    """
    execution_context = call.execution_context
    if call.status != "ok" or not has_cap(call.tool_name, JUDGE, call.agent.tool_caps):
        return
    def _arg(role: str) -> str:
        names = arg_role(call.tool_name, role, call.agent.tool_caps)
        return str(call.arguments.get(names[0], "") or "").strip() if names else ""

    verdict = _arg("verdict").lower()
    settled = apply_verdict(
        verdict, _arg("verdict_reason"), _arg("verdict_scope"), execution_context)
    for run in settled:
        call_id = run.get("call_id")
        if call_id:
            emit({"type": "verdict", "id": call_id, "verdict": verdict})

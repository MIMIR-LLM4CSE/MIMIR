"""Statements about evidence: verdicts on runs, and denials a later call clears."""
from __future__ import annotations

from __future__ import annotations
from ...event_sink import emit
from ..verdict import apply_verdict
from ...context.capabilities import (
    JUDGE,
    arg_role,
    has_cap,
)
from .call import Call


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
    denied_calls = execution_context.denied_tool_calls
    execution_context.denied_tool_calls = [
        item for item in denied_calls
        if not (
            item.get("tool") == call.tool_name
            and item.get("path") == resolved_path
        )
    ]
    if not execution_context.denied_tool_calls:
        execution_context.nudge_counts["denial"] = 0


def verdict_tool(call: Call) -> None:
    """The model judged a run's output: settle the run it speaks for.

    The judging tool is found by capability and read through its declared arg-roles, so
    neither the tool's name nor its parameter names appear here — the server owns both.
    Its own result carries nothing: what the call *means* is state on the blackboard,
    which is what this writes.

    Settled runs are announced per call id so the UI can badge the run's own row; a
    statement that settled nothing (an unqualified ``pass`` over runs bearing on
    different files) announces nothing, and the nudge asks for the missing scope.
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

"""The anti-thrashing stop: a file whose validation failed ``VALIDATION_RETRY_BUDGET`` times.

The ``validate`` state blocks nothing (POLICY.md, "Workflow state machine").
"""
from __future__ import annotations

from ...context.context_view import ContextView

from ...tools.capabilities import EDIT, has_cap, target_path_arg
from ...config.limits import VALIDATION_RETRY_BUDGET
from ...protocol.tool_result import json_error_payload
from ...core.ports import AgentPort


def check_state_machine_guard(
    agent: AgentPort,
    tool_name: str,
    arguments: dict,
    execution_context: ContextView | None,
) -> str | None:
    if execution_context is None:
        return None

    state = execution_context.workflow_state
    path = agent.normalize_workspace_path(
        target_path_arg(tool_name, arguments, agent.tool_caps))
    is_code_target = bool(path and agent.is_code_filepath(path))

    if not (has_cap(tool_name, EDIT, agent.tool_caps) and is_code_target and path):
        return None

    fail_counts = execution_context.checks.validation_fail_count_by_file
    failure_count = int(fail_counts.get(path, 0))

    # Hard stop only for the specific file that exhausted its validation retry budget.
    if state == "edit" and failure_count >= VALIDATION_RETRY_BUDGET:
        return json_error_payload(
            f"Workflow state guard blocked '{tool_name}' after repeated validation failures.",
            hint=(
                f"Validation failed {failure_count} times for this file. "
                "Stop retrying broad edits on this same file; either propose a smaller, testable fix "
                "or conclude with a clear explanation of the remaining risk."
            ),
            state=state,
            tool=tool_name,
            path=path,
        )

    # No branch for the `validate` state: steering the model back toward pending
    # validation is the validation nudge's job (advisory, dial-tunable), not a block's.
    return None

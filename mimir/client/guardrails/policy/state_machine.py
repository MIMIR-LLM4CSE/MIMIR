"""Workflow-state guard: the anti-thrashing hard stop on a file that keeps failing.

A reality check rather than a ritual — a file whose validation has failed
``VALIDATION_RETRY_BUDGET`` times is one the model is patching blind, and each further
broad edit makes the change harder to reason about, not closer to passing.

The ``validate`` state deliberately blocks nothing; steering the model back toward
pending validation is the validation nudge's job. See POLICY.md → *Workflow State
Machine* for why the branch that did block there was removed.
"""
from __future__ import annotations

from typing import Any

from ...context import bootstrap_state_context
from ...context.capabilities import EDIT, has_cap
from ...config.constants import VALIDATION_RETRY_BUDGET


def check_state_machine_guard(
    agent: Any,
    tool_name: str,
    arguments: dict,
    execution_context: dict | None,
) -> str | None:
    execution_context = bootstrap_state_context(execution_context)
    if execution_context is None:
        return None

    state = execution_context.get("workflow_state", "discover")
    path = agent._normalize_workspace_path(arguments.get("path") or arguments.get("filepath"))
    is_code_target = bool(path and agent._is_code_filepath(path))

    if not (has_cap(tool_name, EDIT, agent.tool_caps) and is_code_target and path):
        return None

    fail_counts = execution_context.get("validation_fail_count_by_file", {})
    failure_count = int(fail_counts.get(path, 0))

    # Hard stop only for the specific file that exhausted its validation retry budget.
    if state == "edit" and failure_count >= VALIDATION_RETRY_BUDGET:
        return agent._json_error_payload(
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

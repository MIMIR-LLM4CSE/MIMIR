"""Write-policy gate: preconditions that block a file mutation.

Each rule guards against a loss (an unread overwrite, a guessed deletion, a patch
retried into the ground); a preferred working order belongs to the nudges.
"""
from __future__ import annotations

import os

from ...context import (
    ensure_execution_context,
    is_known_to_exist,
    was_read,
)
from ...tools.capabilities import EDIT, OVERWRITE, REMOVE, has_cap, target_path_arg
from ...config.limits import REPEATED_EDIT_FAILURE_LIMIT
from ...protocol.tool_result import json_error_payload
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort


def has_delete_context(path: str, execution_context: ExecutionContext) -> bool:
    """Require stronger evidence before allowing deletion.

    `checked_paths` proves only that a check was attempted, not that the file really exists.
    For destructive actions, require explicit existence/read evidence plus parent-dir context.
    """
    parent = os.path.dirname(path) or "."

    parent_known = parent in execution_context.discovery.inspected_dirs
    return is_known_to_exist(execution_context, path) and parent_known


# The one violation that may come from a mistyped target rather than from the policy:
# the caller can offer to clarify the path for this kind and only this kind.
UNCONFIRMED_PATH = "unconfirmed_path"


def write_policy_violation(tool_name: str, path: str, detail: str | None = None,
                           *, kind: str = "policy") -> str:
    return json_error_payload(
        f"Write policy blocked tool '{tool_name}' for path '{path}'.",
        hint=detail or (
            "Inspect the workspace structure first (list/search), then confirm the "
            "exact target path (existence check / read) before retrying."
        ),
        violation_kind=kind,
    )


def check_write_policy(
    agent: AgentPort,
    tool_name: str,
    arguments: dict,
    execution_context: ExecutionContext | None,
) -> str | None:
    execution_context = ensure_execution_context(execution_context)

    if execution_context is None or not agent.is_write_tool(tool_name):
        return None

    path = agent.normalize_workspace_path(
        target_path_arg(tool_name, arguments, agent.tool_caps))
    if not path:
        return None

    is_code_target = agent.is_code_filepath(path)
    failures = execution_context.edits.edit_failures.get(path)
    repeated_failures = failures.identical if failures else 0

    if has_cap(tool_name, REMOVE, agent.tool_caps):
        has_direct_context = has_delete_context(path, execution_context)
        if not has_direct_context:
            detail = (
                "Confirm the exact target exists and inspect it (existence check, ranged "
                "read, or directory listing) before deleting it."
            )
            return write_policy_violation(tool_name, path, detail=detail,
                                          kind=UNCONFIRMED_PATH)
        return None

    if has_cap(tool_name, OVERWRITE, agent.tool_caps):
        # "Known to exist but never read" — the two facts are different. What this
        # refuses is rewriting a file nobody looked at; it asks for a read, not for the
        # whole file, since reading is targeted by design.
        if is_known_to_exist(execution_context, path) and not was_read(
            execution_context, path
        ):
            detail = (
                "Overwriting an existing file requires reading it first so the rewrite is "
                "grounded in the current content."
            )
            return write_policy_violation(tool_name, path, detail=detail)

        return None

    if has_cap(tool_name, EDIT, agent.tool_caps) and is_code_target:
        if repeated_failures >= REPEATED_EDIT_FAILURE_LIMIT:
            detail = (
                "Detected repeated identical failed edit attempts. Stop retrying the same patch; "
                "read the file again, choose a different anchor strategy, rewrite the section more broadly, "
                "or conclude with a clear explanation of what failed."
            )
            return write_policy_violation(tool_name, path, detail=detail)

    return None

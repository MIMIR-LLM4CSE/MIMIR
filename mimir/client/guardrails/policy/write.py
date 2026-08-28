"""Write-policy gate: preconditions that BLOCK a file mutation.

The hard write guard (read-before-overwrite, delete evidence, anti-thrashing).
Split out of the former ``runtime.py``; the observation layer that populates the
execution_context it reads lives in ``guardrails/observations.py``.

Every rule here guards against a *loss* — an unread file overwritten, a file deleted
on a guess, a patch retried into the ground. A rule that only enforces a preferred
working order does not belong in this module: it blocks reversible work, and being a
hard guard it would sit outside the enforcement dial that exists to tune exactly that.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ...context import (
    backfill_execution_context,
    ensure_execution_context,
    is_known_to_exist,
    was_read,
)
from ...context.capabilities import (
    EDIT,
    OVERWRITE,
    REMOVE,
    has_cap,
)
from ...config.constants import REPEATED_EDIT_FAILURE_LIMIT


def has_delete_context(path: str, execution_context: dict[str, Any]) -> bool:
    """Require stronger evidence before allowing deletion.

    `checked_paths` proves only that a check was attempted, not that the file really exists.
    For destructive actions, require explicit existence/read evidence plus parent-dir context.
    """
    parent = os.path.dirname(path) or "."
    execution_context = backfill_execution_context(execution_context)

    parent_known = parent in execution_context["inspected_dirs"]
    return is_known_to_exist(execution_context, path) and parent_known


def write_policy_violation(tool_name: str, path: str, detail: str | None = None) -> str:
    payload = {
        "status": "error",
        "error": f"Write policy blocked tool '{tool_name}' for path '{path}'.",
        "hint": detail or (
            "Inspect the workspace structure first (list/search), then confirm the "
            "exact target path (existence check / read) before retrying."
        ),
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


def check_write_policy(
    agent: Any,
    tool_name: str,
    arguments: dict,
    execution_context: dict[str, Any] | None,
) -> str | None:
    execution_context = ensure_execution_context(execution_context)
    execution_context = backfill_execution_context(execution_context)

    if execution_context is None or not agent._is_write_tool(tool_name):
        return None

    path = agent._normalize_workspace_path(arguments.get("path"))
    if not path:
        return None

    is_code_target = agent._is_code_filepath(path)
    repeated_failures = execution_context["edit_loop_state"].get(path, (None, 0))[1]

    if has_cap(tool_name, REMOVE, agent.tool_caps):
        has_direct_context = has_delete_context(path, execution_context)
        if not has_direct_context:
            detail = (
                "Confirm the exact target exists and inspect it (existence check, ranged "
                "read, or directory listing) before deleting it."
            )
            return write_policy_violation(tool_name, path, detail=detail)
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

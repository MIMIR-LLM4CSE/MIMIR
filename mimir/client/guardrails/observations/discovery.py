"""What searches, listings and existence checks tell the guardrails."""
from __future__ import annotations

from __future__ import annotations
from ..workflow import (
    set_workflow_state,
)
from ...context.capabilities import (
    CANDIDATE_SEARCH,
    CHECK_EXISTENCE,
    INSPECT_DIR,
    SEARCH,
    has_cap,
)
from ...context.paths import parent_path
from .call import Call
from .files import register_known_path


def search_flags(call: Call) -> None:
    execution_context = call.execution_context
    if not (has_cap(call.tool_name, SEARCH, call.agent.tool_caps) and call.status == "ok"):
        return
    execution_context.searched = True
    execution_context.search_tool_calls += 1


def discover_transition(call: Call) -> None:
    """Leave discover state once any tool returns concrete results."""
    execution_context = call.execution_context
    if execution_context.workflow_state != "discover":
        return
    search_has_results = bool(
        call.payload.get("matches")
        or call.payload.get("files")
        or call.payload.get("results")
        or call.payload.get("lines_returned")
        or call.payload.get("tree")
        or call.payload.get("entries")
    )
    if search_has_results:
        set_workflow_state(execution_context, "edit")


def candidates(call: Call) -> None:
    execution_context = call.execution_context
    if not has_cap(call.tool_name, CANDIDATE_SEARCH, call.agent.tool_caps):
        return
    for item in call.payload.get("matches", []):
        if not isinstance(item, dict):
            continue
        candidate = call.agent.normalize_workspace_path(item.get("path") or item.get("file"))
        if not candidate:
            continue
        register_known_path(execution_context, candidate)
        candidate_parent = parent_path(candidate)
        by_dir = execution_context.similar_candidates_by_dir
        by_dir.setdefault(candidate_parent, set()).add(candidate)


def dir_inspect(call: Call) -> None:
    execution_context = call.execution_context
    if not (has_cap(call.tool_name, INSPECT_DIR, call.agent.tool_caps) and call.status == "ok"):
        return
    if call.path:
        execution_context.inspected_dirs.add(call.path)
    listed_dir = call.agent.normalize_workspace_path(
        call.payload.get("path") or call.payload.get("parent"))
    if listed_dir:
        execution_context.inspected_dirs.add(listed_dir)


def existence_check(call: Call) -> None:
    execution_context = call.execution_context
    if not (has_cap(call.tool_name, CHECK_EXISTENCE, call.agent.tool_caps)
            and call.arguments.get("path")):
        return
    checked = call.agent.normalize_workspace_path(call.arguments.get("path"))
    if not checked:
        return
    if call.status == "ok" or call.payload.get("exists") is not None:
        execution_context.checked_paths.add(checked)
    if call.payload.get("exists") is True:
        register_known_path(execution_context, checked)

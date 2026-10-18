"""What the task checklist tells the guardrails: its flags and the declared edit set."""
from __future__ import annotations

from __future__ import annotations
import re as _re
from ...context import (
    SOURCE_FILE_EXTENSIONS,
)
from ...tools.capabilities import TASK_PLANNING, arg_role, names_with_cap
from .call import Call


# Source-file path regex derived from the canonical extension set so it stays in sync
# (and automatically covers CUDA/Fortran/… as SOURCE_FILE_EXTENSIONS evolves). Longest
# extensions first so e.g. ".f90" is preferred over ".f".
_SOURCE_EXT_ALTERNATION = "|".join(
    _re.escape(ext.lstrip("."))
    for ext in sorted(SOURCE_FILE_EXTENSIONS, key=len, reverse=True)
)


_SOURCE_FILE_PATH_RE = _re.compile(rf"[\w./\-]+\.(?:{_SOURCE_EXT_ALTERNATION})\b")


def todo_flags(call: Call) -> None:
    """Flag that a task plan was recorded. The checklist (ordered steps) and the prose
    rationale are the same TASK_PLANNING capability, told apart by the `plan_steps`
    arg-role — the checklist carries the steps, the rationale does not."""
    execution_context = call.execution_context
    planning = names_with_cap(TASK_PLANNING, call.agent.tool_caps)
    if call.status != "ok" or call.tool_name not in planning:
        return
    if arg_role(call.tool_name, "plan_steps", call.agent.tool_caps):
        execution_context.plan.todo_written = True
    else:
        execution_context.plan.plan_written = True


def declared_edit_set(call: Call) -> None:
    """Replace the declared edit set with the source paths a new checklist names.

    Only the checklist tool (``plan_steps`` arg-role) declares; a checklist naming no file
    empties the set.
    """
    execution_context = call.execution_context
    if call.status != "ok":
        return
    step_args = arg_role(call.tool_name, "plan_steps", call.agent.tool_caps)
    if not step_args:
        return
    raw_steps = next((call.arguments.get(a) for a in step_args if call.arguments.get(a)), "")
    if isinstance(raw_steps, list):
        raw_steps = " ".join(str(s) for s in raw_steps)
    declared = {
        candidate
        for match in _SOURCE_FILE_PATH_RE.finditer(str(raw_steps))
        if (candidate := call.agent.normalize_workspace_path(match.group(0)))
    }
    # Mutated in place: other modules hold a reference to this set.
    execution_context.edits.declared_edit_set.clear()
    execution_context.edits.declared_edit_set.update(declared)

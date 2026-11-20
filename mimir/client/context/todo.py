"""The task checklist as the client reads it: the steps on disk (``shared.checklist``),
each tagged optional or not, and ready or not when a deps sidecar names what waits on what."""
from __future__ import annotations
from typing import Any

import logging
import re

from ...shared.checklist import read_checklist, read_deps, write_checklist, write_deps

logger = logging.getLogger(__name__)

# A step the plan itself marked aspirational ("(optional) convergence study").
# Without this, such a step is stored as a plain `- [ ]` like any other and, since
# nothing gates on unchecked items, vanishes silently — which is exactly how a
# planned convergence study disappeared between plan and delivery. Tagged here so
# the completion ledger can report it separately without it blocking anything.
_OPTIONAL_STEP_RE = re.compile(r"^\s*[\(\[]?optional[\)\]]?\s*[:\-–]?\s+", re.IGNORECASE)


def load_todo_items(todo_file: str) -> list[dict[str, Any]]:
    """The checklist in *todo_file*, [] on any error.

    Each item says whether it is ``optional``; with a deps sidecar, also whether it is
    ``ready`` (not done, everything it waits on done).
    """
    items = read_checklist(todo_file)
    for item in items:
        item["optional"] = bool(_OPTIONAL_STEP_RE.match(item["text"]))

    deps = read_deps(todo_file)
    if deps:
        try:
            for item in items:
                if item["done"]:
                    item["ready"] = False  # already done, not actionable
                    continue
                idx: int = item["index"]
                step_deps = deps[idx] if idx < len(deps) else []
                item["ready"] = all(
                    items[d]["done"] for d in step_deps if d < len(items)
                )
        except Exception:
            logger.debug("load_todo_items: a best-effort step failed and was skipped", exc_info=True)

    return items


def write_todo_items(todo_file: str, todos: list[dict[str, Any]],
                     todo_deps: list[list[int]] | None = None) -> None:
    """Write *todos* and their deps sidecar as the checklist in *todo_file*; none removes both.

    Raises on failure.
    """
    write_checklist(todo_file, todos)
    write_deps(todo_file, todo_deps if todos else None)

"""The task checklist as the agent keeps it on disk: ``todo_list.md``, plus an optional
``todo_deps.json`` sidecar naming which step waits on which."""
from __future__ import annotations
from typing import Any

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

# A step the plan itself marked aspirational ("(optional) convergence study").
# Without this, such a step is stored as a plain `- [ ]` like any other and, since
# nothing gates on unchecked items, vanishes silently — which is exactly how a
# planned convergence study disappeared between plan and delivery. Tagged here so
# the completion ledger can report it separately without it blocking anything.
_OPTIONAL_STEP_RE = re.compile(r"^\s*[\(\[]?optional[\)\]]?\s*[:\-–]?\s+", re.IGNORECASE)


def load_todo_items(todo_file: str) -> list[dict[str, Any]]:
    """Read the current todo list from todo_list.md, return [] on any error.

    If a todo_deps.json sidecar exists in the same directory, each item is
    annotated with a ``ready`` boolean so the context injection can inform
    the model which steps are unblocked.
    """
    _checkbox = re.compile(r"^- \[([ x])\] (.+)$")
    try:
        with open(todo_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    items: list[dict[str, Any]] = []
    for line in lines:
        m = _checkbox.match(line.rstrip("\n"))
        if m:
            text = m.group(2)
            items.append({
                "index": len(items),
                "text": text,
                "done": m.group(1) == "x",
                "optional": bool(_OPTIONAL_STEP_RE.match(text)),
            })

    # Annotate with DAG ready status if deps sidecar exists.
    deps = load_todo_deps(todo_file)
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
            pass  # deps annotation is best-effort

    return items


def load_todo_deps(todo_file: str) -> list[list[int]]:
    """The ``todo_deps.json`` sidecar beside *todo_file*: per step, the steps it waits
    on. [] when there is none or it cannot be read."""
    deps_file = os.path.join(os.path.dirname(todo_file), "todo_deps.json")
    try:
        with open(deps_file, "r", encoding="utf-8") as f:
            deps = json.load(f)
    except (OSError, ValueError):
        return []
    return deps if isinstance(deps, list) else []

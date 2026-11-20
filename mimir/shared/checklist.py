"""The task checklist on disk, one format for both ends.

``todo_list.md`` holds one ``- [ ] step`` / ``- [x] step`` line per step; an optional
``todo_deps.json`` beside it gives, per step, the steps it waits on. The todo server writes
them and the client reads them. :func:`render_checklist` is the one rendering the model
sees, in the system message at the start of a turn and in a todo tool's reply within it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEPS_FILENAME = "todo_deps.json"

_CHECKBOX_RE = re.compile(r"^- \[([ x])\] (.+)$")


def read_checklist(todo_file: str) -> list[dict[str, Any]]:
    """The steps in *todo_file* as ``{"index", "text", "done"}``; [] when it cannot be read."""
    try:
        with open(todo_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    items: list[dict[str, Any]] = []
    for line in lines:
        m = _CHECKBOX_RE.match(line.rstrip("\n"))
        if m:
            items.append({"index": len(items), "text": m.group(2), "done": m.group(1) == "x"})
    return items


def write_checklist(todo_file: str, items: list[dict[str, Any]]) -> None:
    """Write *items* as the checklist in *todo_file*; none removes the file. Raises on failure."""
    if not items:
        try:
            os.remove(todo_file)
        except FileNotFoundError:
            pass
        return
    os.makedirs(os.path.dirname(todo_file), exist_ok=True)
    with open(todo_file, "w", encoding="utf-8") as f:
        f.write("".join(f"- [{'x' if it.get('done') else ' '}] {it.get('text', '')}\n"
                        for it in items))


def _deps_file(todo_file: str) -> str:
    return os.path.join(os.path.dirname(todo_file), DEPS_FILENAME)


def read_deps(todo_file: str) -> list[list[int]]:
    """Per step, the steps it waits on, from the sidecar beside *todo_file*; [] if none."""
    try:
        with open(_deps_file(todo_file), "r", encoding="utf-8") as f:
            deps = json.load(f)
    except (OSError, ValueError):
        return []
    return deps if isinstance(deps, list) else []


def write_deps(todo_file: str, deps: list[list[int]] | None) -> None:
    """Write the sidecar beside *todo_file*; None or empty removes it."""
    path = _deps_file(todo_file)
    if not deps:
        try:
            os.remove(path)
        except OSError:
            pass
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(deps, f)


def render_checklist(items: list[dict[str, Any]]) -> str:
    """``[x] text`` / ``[ ] text``, one line per step."""
    return "\n".join(f"[{'x' if it.get('done') else ' '}] {it['text']}" for it in items)

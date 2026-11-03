"""The R&D requests put to the agent from the code map (``proxy_bench/requests.json``):
what to try on which code, in study or in try, and where the session stood when it was
opened, so every variant of it is measured from the same tree. A run belongs to a request
through its axis: ``[R3]`` or ``[R3.2]`` for its second variant."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone

from ....shared.atomic import write_json
from .layout import read_json, requests_file
from .store import file_lock

MODES = ("study", "try")
KINDS = ("method", "algorithm", "external", "parallel", "layout", "precision", "refactor", "other")
STATUSES = ("open", "studying", "proposed", "trying", "done", "dropped")

# `[R3]` or `[R3.2]` at the head of an axis: the request, and the variant when there is one.
_AXIS_RE = re.compile(r"^\s*\[R(\d+)(?:\.(\d+))?\]")


def load() -> list[dict]:
    """Every request, oldest first; none when the file is missing or unreadable."""
    found = read_json(requests_file(), [])
    return found if isinstance(found, list) else []


def _lock() -> str:
    return requests_file() + ".lock"


def add(fields: dict) -> dict:
    """Record a new request under the next number, and return it."""
    with file_lock(_lock()):
        rows = load()
        n = 1 + max((int(r["id"][1:]) for r in rows if re.fullmatch(r"R\d+", str(r.get("id", "")))), default=0)
        row = {"id": f"R{n}", "ts": datetime.now(timezone.utc).isoformat(), "status": "open", **fields}
        rows.append(row)
        write_json(requests_file(), rows, indent=2)
    return row


def get(request_id: str) -> dict | None:
    """The request *request_id* (``R3``, or ``3``), or None."""
    want = normalise(request_id)
    return next((r for r in load() if r.get("id") == want), None)


def update(request_id: str, changes: dict) -> dict | None:
    """Apply *changes* to the request and return it; None when there is no such request."""
    want = normalise(request_id)
    with file_lock(_lock()):
        rows = load()
        row = next((r for r in rows if r.get("id") == want), None)
        if row is None:
            return None
        row.update(changes)
        row["updated"] = datetime.now(timezone.utc).isoformat()
        write_json(requests_file(), rows, indent=2)
    return row


def normalise(request_id: str) -> str:
    """``3``, ``r3`` and ``R3`` all name ``R3``."""
    bare = str(request_id or "").strip().upper().lstrip("R")
    return f"R{bare}" if bare.isdigit() else str(request_id or "").strip()


def axis_of(axis: str) -> tuple[str, int | None] | None:
    """The request an axis names, and its variant (1-based) when it names one; else None."""
    m = _AXIS_RE.match(axis or "")
    if not m:
        return None
    return f"R{m.group(1)}", int(m.group(2)) if m.group(2) else None


def exists() -> bool:
    return os.path.exists(requests_file())


def is_refactor(axis: str) -> bool:
    """Whether a run's axis names a refactor request: the ratchet keeps it when neutral."""
    tag = axis_of(axis)
    row = get(tag[0]) if tag else None
    return bool(row and row.get("kind") == "refactor")

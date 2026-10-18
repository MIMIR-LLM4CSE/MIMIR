"""The file a call touched, and where, for the row's clickable name (``target`` on ``tool_result``).

Found by argument names and result keys; sent only for a successful call on a file.
"""

from __future__ import annotations

import os
from typing import Any

from ..protocol.tool_result import parse_tool_payload
from .capabilities import ToolCaps, target_path_arg

# Result keys carrying the span a call covered, in priority order: an edit reports
# where it landed in the file it produced, a read the range it returned.
_SPAN_KEYS = (("new_start_line", "new_end_line"), ("start_line", "end_line"))


def _as_line(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _path_arg(name: str, args: dict[str, Any], tool_caps: dict[str, ToolCaps] | None) -> str | None:
    raw = target_path_arg(name, args, tool_caps)
    return raw if raw and os.path.isabs(raw) else None


def result_span(result_text: str) -> dict[str, int]:
    """``{line, end_line}`` the result reports, or ``{}``. A pure deletion, whose
    span ends before it starts, keeps only the line where the text was."""
    payload = parse_tool_payload(result_text) if isinstance(result_text, str) else None
    if not payload:
        return {}
    for start_key, end_key in _SPAN_KEYS:
        start = _as_line(payload.get(start_key))
        if start is None:
            continue
        end = _as_line(payload.get(end_key))
        return {"line": start, "end_line": end} if end and end >= start else {"line": start}
    return {}


def file_target(name: str, args: dict[str, Any], result_text: str,
                tool_caps: dict[str, ToolCaps] | None = None) -> dict[str, Any] | None:
    """The ``target`` of a successful call, or None when it touched no file."""
    if not isinstance(args, dict):
        return None
    path = _path_arg(name, args, tool_caps)
    if not path or not os.path.isfile(path):
        return None
    target: dict[str, Any] = {"path": path, "name": os.path.basename(path)}
    target.update(result_span(result_text))
    return target

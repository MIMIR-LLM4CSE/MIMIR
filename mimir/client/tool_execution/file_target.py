"""The file a tool call touched, and where, for the row's clickable file name.

A row shows the file name alone (see ``tool_status_messages.shorten_display_args``),
so the UI cannot open the file from what it displays. This rides on the
``tool_result`` event as ``target: {path, name, line?, end_line?}``: the absolute
path, the name the row shows, and the lines to select.

Found by argument name and result keys, never by tool name, like the shortening it
undoes. Sent only for a call that succeeded on an existing file: a failed read or
edit says nothing reliable about the file, and a directory has no lines to show.
"""

from __future__ import annotations

import os
from typing import Any

from .formatter import parse_tool_payload
from .tool_status_messages import _PATH_KEYS

# Result keys carrying the span a call covered, in priority order: an edit reports
# where it landed in the file it produced, a read the range it returned.
_SPAN_KEYS = (("new_start_line", "new_end_line"), ("start_line", "end_line"))


def _as_line(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _path_arg(name: str, args: dict, tool_caps) -> str | None:
    from ..context.capabilities import arg_role
    keys = arg_role(name, "path", tool_caps) or _PATH_KEYS
    for key in keys:
        val = args.get(key)
        if isinstance(val, str) and os.path.isabs(val.strip()):
            return val.strip()
    return None


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


def file_target(name: str, args: dict, result_text: str, tool_caps=None) -> dict | None:
    """The ``target`` of a successful call, or None when it touched no file."""
    if not isinstance(args, dict):
        return None
    path = _path_arg(name, args, tool_caps)
    if not path or not os.path.isfile(path):
        return None
    target: dict[str, Any] = {"path": path, "name": os.path.basename(path)}
    target.update(result_span(result_text))
    return target

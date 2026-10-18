"""The terminal-panel preview of an exec-shaped result (``returncode`` + ``stdout``/``stderr``).

Detected by shape, never by tool name. Attached to ``tool_result`` as ``exec``; streams
are ANSI-stripped and clipped for display only.
"""

from __future__ import annotations

import re
from typing import Any

from ..protocol.tool_result import parse_tool_payload
from ..context.tool_display import COMMAND_KEYS

# CSI / OSC escape sequences (colors, cursor movement) — noise in an HTML panel.
_ANSI_RE = re.compile(r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\))")

# Per-stream wire budget. stdout keeps its TAIL (the end of a run — final
# results, tracebacks — matters most); stderr keeps its HEAD (the first error
# usually causes the rest).
_MAX_STREAM_LINES = 200
_MAX_STREAM_CHARS = 16_000
_MAX_COMMAND_CHARS = 2_000


def _clip_stream(text: str, keep: str) -> tuple[str, bool]:
    """ANSI-strip and clip *text* to the wire budget, keeping head or tail."""
    text = _ANSI_RE.sub("", text)
    clipped = False
    lines = text.splitlines()
    if len(lines) > _MAX_STREAM_LINES:
        omitted = len(lines) - _MAX_STREAM_LINES
        if keep == "tail":
            lines = [f"… (+{omitted} earlier lines omitted)"] + lines[-_MAX_STREAM_LINES:]
        else:
            lines = lines[:_MAX_STREAM_LINES] + [f"… (+{omitted} more lines omitted)"]
        text = "\n".join(lines)
        clipped = True
    if len(text) > _MAX_STREAM_CHARS:
        if keep == "tail":
            text = "…" + text[-_MAX_STREAM_CHARS:]
        else:
            text = text[:_MAX_STREAM_CHARS] + "…"
        clipped = True
    return text, clipped


def _command_from_args(arguments: dict | None) -> str | None:
    """The command body, from whichever argument carries it.

    Same key priority as the ``tool_call`` ``detail`` preview, but the full text:
    the panel shows real input, clipped only at the wire budget.
    """
    if not isinstance(arguments, dict):
        return None
    for key in COMMAND_KEYS:
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            command = val.strip()
            if len(command) > _MAX_COMMAND_CHARS:
                command = command[:_MAX_COMMAND_CHARS] + "…"
            return command
    return None


def exec_input_preview(arguments: dict | None) -> dict[str, Any] | None:
    """The IN half of the panel, sent on ``tool_call`` so it shows while the run is going.

    Same ``exec`` shape with empty streams; ``None`` when there is no command to show.
    """
    command = _command_from_args(arguments)
    if not command:
        return None
    return {"command": command, "stdout": "", "stderr": ""}


def extract_exec_preview(result_text: str, arguments: dict | None) -> dict[str, Any] | None:
    """The ``exec`` display object for a finished run, or for a run moved to the background.

    ``None`` for anything else, which keeps the row a plain summary.
    """
    payload = parse_tool_payload(result_text)
    if payload is None:
        return None
    running = bool(payload.get("background_job"))
    if "returncode" not in payload and not running:
        return None
    if "stdout" not in payload and "stderr" not in payload:
        return None

    returncode: int | None = None
    if "returncode" in payload:
        try:
            returncode = int(payload["returncode"])
        except (TypeError, ValueError):
            return None
    elif not running:
        return None

    stdout, out_clipped = _clip_stream(str(payload.get("stdout") or ""), keep="tail")
    stderr, err_clipped = _clip_stream(str(payload.get("stderr") or ""), keep="head")

    info: dict[str, Any] = {
        "stdout": stdout,
        "stderr": stderr,
    }
    if returncode is not None:
        info["returncode"] = returncode
    if running:
        info["running"] = True
        job_key = (payload.get("background_job") or {}).get("job_key")
        if isinstance(job_key, str) and job_key:
            info["job_key"] = job_key

    command = _command_from_args(arguments)
    if command:
        info["command"] = command

    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        info["cwd"] = cwd

    if payload.get("truncated") or out_clipped or err_clipped:
        info["truncated"] = True
    return info

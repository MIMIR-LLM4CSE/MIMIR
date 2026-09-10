"""Structured exec-output preview for command-like tool results.

A tool result is "exec-shaped" when its JSON payload carries a ``returncode``
plus ``stdout``/``stderr`` — the envelope every command runner (shell, code
runner, compiler) returns. Detection is by payload *shape*, never by tool name,
so any current or future execution tool gets the terminal panel in the UI for
free, and non-exec tools stay summary-only.

The extracted preview is attached to the ``tool_result`` event as an ``exec``
object: ``{command?, stdout, stderr, returncode, cwd?, truncated?}``. Streams
are ANSI-stripped and clipped for the wire (the full text still reaches the
model through history; this is a display copy only).
"""

from __future__ import annotations

import re
from typing import Any

from .formatter import parse_tool_payload
from .tool_status_messages import _COMMAND_KEYS

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


def extract_exec_preview(result_text: str, arguments: dict | None) -> dict[str, Any] | None:
    """Build the ``exec`` display object for an exec-shaped tool result.

    Two shapes qualify. A finished run carries a ``returncode`` alongside its
    streams. A run the user moved to the background carries no returncode — it has
    not produced one — but does carry a ``background_job`` handle and whatever it had
    printed by then; that one previews as ``running``, and the panel shows the output
    so far instead of an exit badge.

    Returns ``None`` for anything that is neither (unparseable result, or no
    returncode/stdout/stderr keys), which is also the signal that the UI should keep
    a plain summary row.
    """
    payload = parse_tool_payload(result_text)
    if payload is None:
        return None
    running = bool(payload.get("background_job"))
    if "returncode" not in payload and not running:
        return None
    if "stdout" not in payload and "stderr" not in payload:
        return None

    returncode = None
    if "returncode" in payload:
        try:
            returncode = int(payload.get("returncode"))
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

    # The command body, from whichever arg carries it (same key priority as the
    # tool_call `detail` preview, but the full text — the panel shows real input).
    if isinstance(arguments, dict):
        for key in _COMMAND_KEYS:
            val = arguments.get(key)
            if isinstance(val, str) and val.strip():
                command = val.strip()
                if len(command) > _MAX_COMMAND_CHARS:
                    command = command[:_MAX_COMMAND_CHARS] + "…"
                info["command"] = command
                break

    cwd = payload.get("cwd")
    if isinstance(cwd, str) and cwd:
        info["cwd"] = cwd

    if payload.get("truncated") or out_clipped or err_clipped:
        info["truncated"] = True
    return info

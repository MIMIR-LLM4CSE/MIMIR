from __future__ import annotations

import json
from typing import Any


def normalize_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize tool-call arguments from Ollama into a dict."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": arguments}
    return {}


# Payload keys whose value is a block of program text — a file's contents, a run's
# output — rather than a datum.
_TEXT_BLOCK_KEYS = ("content", "stdout", "stderr")
# Names the manifest under, and where the blocks are put back on the way in.
_MANIFEST_KEY = "text_blocks"


def _block_marker(name: str) -> str:
    return f"--- {name} ---"


def _render_payload(payload: Any) -> str:
    """Pretty-print *payload*, with any block of program text moved out of the JSON.

    Text inside a JSON string is escaped text: every newline becomes a literal ``\\n``,
    every quote and backslash doubles, and two hundred lines arrive on one. The model is
    asked to copy anchors *verbatim* from what it read, so the escaping is not merely
    unreadable — an anchor copied from it cannot match the file, and the edit fails for
    a reason nothing in the reply explains. A traceback or a compiler error reaches it
    the same way, folded onto a single line.

    The envelope keeps a manifest of ``{name: length}`` so :func:`parse_tool_payload`
    can put the blocks back exactly, by length rather than by scanning for the marker.
    That is what lets every consumer go on reading ``payload["stdout"]`` while the model
    reads real text, and it is immune both to output that happens to contain a marker
    and to the annotations the executor appends after everything.
    """
    if not isinstance(payload, dict):
        return json.dumps(payload, indent=2, ensure_ascii=False)
    blocks = {
        key: payload[key] for key in _TEXT_BLOCK_KEYS
        if isinstance(payload.get(key), str) and payload[key]
    }
    if not blocks:
        return json.dumps(payload, indent=2, ensure_ascii=False)

    envelope = {k: v for k, v in payload.items() if k not in blocks}
    envelope[_MANIFEST_KEY] = {name: len(text) for name, text in blocks.items()}
    rendered = json.dumps(envelope, indent=2, ensure_ascii=False)
    for name, text in blocks.items():
        rendered += f"\n\n{_block_marker(name)}\n{text}"
    return rendered


def normalize_tool_content(result: Any) -> str:
    """Convert an MCP tool result into a text payload for Ollama."""
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            text = block.text
            try:
                parts.append(_render_payload(json.loads(text)))
            except json.JSONDecodeError:
                parts.append(text)
        else:
            parts.append(str(block))
    return "\n".join(parts)


def truncate_text(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def json_error_payload(message: str, hint: str = "", **extra: Any) -> str:
    payload: dict[str, Any] = {
        "status": "error",
        "error": message,
    }
    if hint:
        payload["hint"] = hint
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def parse_tool_payload(result_text: str) -> dict[str, Any] | None:
    """The leading JSON object of a tool result, with its text blocks put back.

    A result is an envelope, then the blocks named in its manifest, then whatever the
    executor appended. Callers see the payload the server built: the split is a wire
    format for the model's benefit, not a change to the contract they read.
    """
    try:
        payload, end = json.JSONDecoder().raw_decode(result_text.lstrip())
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    manifest = payload.pop(_MANIFEST_KEY, None)
    if isinstance(manifest, dict):
        _restore_text_blocks(payload, manifest, result_text.lstrip()[end:])
    return payload


def _restore_text_blocks(payload: dict, manifest: dict, tail: str) -> None:
    """Slice *tail* back into the named blocks, by length from each marker."""
    cursor = 0
    for name, length in manifest.items():
        if not isinstance(length, int) or length < 0:
            continue
        marker = tail.find(_block_marker(name), cursor)
        if marker < 0:
            continue
        start = marker + len(_block_marker(name)) + 1  # past the marker's newline
        payload[name] = tail[start:start + length]
        cursor = start + length

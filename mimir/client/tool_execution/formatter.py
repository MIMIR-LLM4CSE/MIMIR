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


def normalize_tool_content(result: Any) -> str:
    """Convert an MCP tool result into a text payload for Ollama."""
    parts: list[str] = []
    for block in result.content:
        if hasattr(block, "text"):
            text = block.text
            try:
                parts.append(json.dumps(json.loads(text), indent=2, ensure_ascii=False))
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
    try:
        payload = json.loads(result_text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None

"""Shared response helpers for MCP servers.

Current response API contract:
- success: {"status": "ok", ...domain fields...}
- error:   {"status": "error", "error": "...", "hint"?: "...", ...}
"""

from __future__ import annotations

from typing import Any

_MAX_ERR_LEN = 4096


def _drop_reserved_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Remove reserved protocol keys from payload overrides."""
    return {
        key: value
        for key, value in data.items()
        if key not in {"status", "error", "hint"}
    }


def _normalize_message(message: Any) -> str:
    text = str(message).strip()
    if not text:
        return "Unknown error."
    # Collapse internal whitespace to keep errors compact and consistent.
    text = " ".join(text.split())
    if len(text) > _MAX_ERR_LEN:
        text = text[:_MAX_ERR_LEN] + " ... [truncated]"
    return text


def _auto_hint_for_error(message: str) -> str:
    lower = message.lower()
    if "outside" in lower and "root" in lower:
        return "Use a path inside the allowed workspace or sandbox root."
    if "not found" in lower or "does not exist" in lower:
        return "Check the path/name and list available resources before retrying."
    if "unsupported" in lower:
        return "Use one of the supported values documented by the tool."
    if "forbidden" in lower or "not allowed" in lower or "denied" in lower:
        return "Use an allowed operation or request explicit approval."
    if "timed out" in lower:
        return "Narrow the scope or increase timeout when the tool supports it."
    if "invalid" in lower or "syntax" in lower:
        return "Fix the input format and try again."
    return ""


def ok(data: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """Build a success payload with stable API shape.

    Backward-compatible usage:
    - ok({"result": 1})
    - ok(result=1, count=3)
    """
    payload: dict[str, Any] = {"status": "ok"}
    if data:
        payload.update(_drop_reserved_fields(data))
    if extra:
        payload.update(_drop_reserved_fields(extra))
    return payload


def err(message: Any, hint: str = "", **extra: Any) -> dict[str, Any]:
    """Build an error payload with stable API shape."""
    normalized_message = _normalize_message(message)
    normalized_hint = str(hint).strip() if hint else ""
    if not normalized_hint:
        normalized_hint = _auto_hint_for_error(normalized_message)

    payload: dict[str, Any] = {
        "status": "error",
        "error": normalized_message,
    }
    if normalized_hint:
        payload["hint"] = normalized_hint
    if extra:
        payload.update(_drop_reserved_fields(extra))
    return payload

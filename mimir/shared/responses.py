"""The shape of every tool answer: ``{"status": "ok", ...}`` or ``{"status": "error", ...}``.

An error always carries ``error`` (the message) and may carry ``hint`` (what to try
next). Other fields are the tool's own; they cannot override these three.
"""

from __future__ import annotations

from typing import Any

_MAX_ERR_LEN = 4096
_RESERVED = frozenset({"status", "error", "hint"})


def _own_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in _RESERVED}


def _tidy(message: Any) -> str:
    """One line, bounded: the model reads it inside a JSON payload."""
    text = " ".join(str(message).split())
    if not text:
        return "Unknown error."
    if len(text) > _MAX_ERR_LEN:
        text = text[:_MAX_ERR_LEN] + " ... [truncated]"
    return text


def _default_hint(message: str) -> str:
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
    """A success: ``ok({"result": 1})`` or ``ok(result=1, count=3)``."""
    payload: dict[str, Any] = {"status": "ok"}
    if data:
        payload.update(_own_fields(data))
    if extra:
        payload.update(_own_fields(extra))
    return payload


def err(message: Any, hint: str = "", *, tidy: bool = True, **extra: Any) -> dict[str, Any]:
    """A failure. With *tidy*, the message is put on one bounded line and a missing
    hint is guessed from it; without, both are kept exactly as given."""
    text = _tidy(message) if tidy else str(message)
    hint = str(hint).strip() if hint else ""
    if not hint and tidy:
        hint = _default_hint(text)
    payload: dict[str, Any] = {"status": "error", "error": text}
    if hint:
        payload["hint"] = hint
    if extra:
        payload.update(_own_fields(extra))
    return payload

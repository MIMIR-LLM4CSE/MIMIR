"""The variant of the prose in force: section and message texts replaced for a run.

The prompt and the nudges read their texts through :func:`text`, which answers the
variant's when one is set and the code's otherwise. A variant is validated and set by
``prompt.variants.use_variant``; with none set, nothing changes.
"""
from __future__ import annotations

from typing import Any

_EMPTY: dict[str, Any] = {"name": "", "sections": {}, "messages": {}}
_active: dict[str, Any] = dict(_EMPTY)


def text(kind: str, key: str, default: str) -> str:
    """The variant's text for *key* among its *kind* (``sections``, ``messages``), else *default*."""
    return str(_active[kind].get(key, default))


def active_name() -> str:
    """The name of the variant in force, or "" for the code's own texts."""
    return str(_active["name"])


def activate(variant: dict[str, Any]) -> dict[str, Any]:
    """Put *variant* in force; return the one it replaces, to restore."""
    global _active
    previous = _active
    _active = {"name": variant.get("name", ""), "sections": dict(variant.get("sections") or {}),
               "messages": dict(variant.get("messages") or {})}
    return previous


def restore(previous: dict[str, Any]) -> None:
    global _active
    _active = previous

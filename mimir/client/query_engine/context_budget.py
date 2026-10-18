"""How much of the model's context window a conversation may fill.

Here rather than in ``config``: the window is the live backend's answer, and config is
the layer nothing above it may be reached from.
"""
from __future__ import annotations

from ..config.limits import (
    CTX_RESERVED_RATIO,
    CTX_TOTAL_COMPACT,
    CTX_TOTAL_FULL,
    INTRA_QUERY_COMPACT_TOKENS,
    TOOL_HISTORY_TOKEN_BUDGET,
)
from .backends.factory import get_backend


def context_budget_for(model: str | None, mode: str = "full",
                       ceiling: int | None = None) -> tuple[int, int, int, int]:
    """``(total, reserved, tool_history_budget, intra_compact_budget)`` in tokens.

    From the backend's real window, else the static per-mode defaults. *ceiling* caps the
    window this agent may fill (a sub-agent's share); it never raises a budget.
    """
    default_total = CTX_TOTAL_FULL if mode == "full" else CTX_TOTAL_COMPACT
    window: int | None = None
    try:
        window = get_backend().context_window(model or "")
    except Exception:
        window = None
    if not window:
        if not ceiling or ceiling >= default_total:
            return default_total, int(default_total * CTX_RESERVED_RATIO), \
                TOOL_HISTORY_TOKEN_BUDGET, INTRA_QUERY_COMPACT_TOKENS
        total = ceiling
    else:
        # Compact mode stays a deliberately small budget even on a large window.
        total = window if mode == "full" else min(window, CTX_TOTAL_COMPACT)
        if ceiling:
            total = min(total, ceiling)
    reserved = int(total * CTX_RESERVED_RATIO)
    usable = max(1, total - reserved)
    return total, reserved, usable, int(usable * 0.8)

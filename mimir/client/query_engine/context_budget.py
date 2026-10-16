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
    """Return (total, reserved, tool_history_budget, intra_compact_budget) tokens.

    The context window is resolved dynamically from the active backend — vLLM's
    reported ``max_model_len`` or Ollama's model context length — so the token
    bar and history trim/compaction match the real window instead of the static
    200K assumption. The trim/compaction thresholds scale to sit just under the
    usable window so they actually engage on small windows.

    Falls back to the static per-mode defaults whenever the window can't be
    determined, so behavior is unchanged in that case.

    ``ceiling`` caps the window this particular agent may fill, below whatever the
    model would allow. It is how a sub-agent is given a share of the window rather
    than all of it: several children on one endpoint each reading a large tree would
    otherwise each budget for the whole thing. A ceiling at or above the resolved
    window changes nothing, so passing one never *raises* a budget.
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

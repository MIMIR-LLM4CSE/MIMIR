"""What a spawn call says about its caller, read off the request ``_meta`` the client sends.

The caller's approval mode, model, mode, the tools it may grant (and which of them
write), and how far the user lets a sub-agent go — and what follows from them: the
grant refused, the budget, the model the child runs on. Absent or unknown values are
read as the weakest.
"""

from __future__ import annotations

import os

from mcp.server.fastmcp import Context

from . import spawn_limits

# The request-_meta key the client sends its approval mode under (client
# guardrails.policy.approval.APPROVAL_MODE_META). A child runs in that mode; absent or
# unknown, it runs in "manual" — the mode that lets nothing sensitive through unasked.
APPROVAL_MODE_META = "mimir/approval_mode"
APPROVAL_MODES = ("manual", "auto", "auto_all")

# The request-_meta key the client sends its current model under (client
# config.models.CALLER_MODEL_META). A child with no model named runs on it.
CALLER_MODEL_META = "mimir/model"

# What the caller may hand over, and the mode it was computed under (client
# config.models.GRANTABLE_TOOLS_META / CALLER_MODE_META). The table is {tool: owning
# server}: the child connects servers, not tools. Absent, only an exploration with no
# named tool runs — a caller that cannot say what it may grant grants nothing.
GRANTABLE_TOOLS_META = "mimir/grantable"
GRANTABLE_WRITERS_META = "mimir/grantable_writers"
CALLER_MODE_META = "mimir/mode"

# How far the user lets a sub-agent go (client config.settings.SUBAGENT_LEVELS,
# config.models.SUBAGENT_LEVEL_META). A rung hands something over, so an absent or
# unknown one is read as the weakest: a child that only reads.
SUBAGENT_LEVELS = ("explore", "build")
DEFAULT_SUBAGENT_LEVEL = SUBAGENT_LEVELS[0]
# The rung's former name, still read: it is in the preferences file of everyone who
# set it, and an unknown rung is read as the weakest — which would take away a right
# the user had granted (client config.settings.SUBAGENT_LEVEL_ALIASES).
_SUBAGENT_LEVEL_ALIASES = {"parallel": "build"}
SUBAGENT_LEVEL_META = "mimir/subagent_level"


def approval_mode(ctx: Context | None) -> str:
    """The approval mode the caller sent with this call, or "manual"."""
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        return "manual"
    mode = getattr(meta, "model_extra", None) or {}
    mode = str(mode.get(APPROVAL_MODE_META) or "").strip().lower()
    return mode if mode in APPROVAL_MODES else "manual"


def model_of(ctx: Context | None) -> str:
    """The model the caller runs on, from this call's _meta, else the spawn-time env.

    The env is only a fallback: it was frozen when this server started, and a
    mid-session model switch never reached it.
    """
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        meta = None
    model = str((getattr(meta, "model_extra", None) or {}).get(CALLER_MODEL_META) or "").strip()
    if model:
        return model
    try:
        from mimir.client.config import DEFAULT_MODEL
    except ImportError:
        DEFAULT_MODEL = ""
    return os.environ.get("MIMIR_DEFAULT_MODEL", "").strip() or DEFAULT_MODEL


def _meta_value(ctx: Context | None, key: str):
    """One key of this call's request _meta, or None."""
    try:
        meta = ctx.request_context.meta if ctx is not None else None
    except (AttributeError, ValueError):
        return None
    return (getattr(meta, "model_extra", None) or {}).get(key)


def grantable(ctx: Context | None) -> dict[str, str]:
    """``{tool: server}`` the caller may grant, from the call's metadata; missing means nothing."""
    raw = _meta_value(ctx, GRANTABLE_TOOLS_META)
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items() if k and v}


def writers(ctx: Context | None) -> set[str]:
    """Which grantable tools change the tree, as the caller's registry reports them."""
    raw = _meta_value(ctx, GRANTABLE_WRITERS_META)
    return {str(x) for x in raw} if isinstance(raw, list) else set()



def subagent_level(ctx: Context | None) -> str:
    """How far the user lets this child go; the weakest rung when unsaid or unknown.

    Another name of a rung ("parallel") resolves here rather than being read as unknown,
    which would take away the right its user granted.
    """
    level = str(_meta_value(ctx, SUBAGENT_LEVEL_META) or "").strip().lower()
    level = _SUBAGENT_LEVEL_ALIASES.get(level, level)
    return level if level in SUBAGENT_LEVELS else DEFAULT_SUBAGENT_LEVEL


def level_allows(level: str, required: str) -> bool:
    level = _SUBAGENT_LEVEL_ALIASES.get(level, level)
    return SUBAGENT_LEVELS.index(level) >= SUBAGENT_LEVELS.index(required)


def grant_refusal(requested: list[str], grantable: dict[str, str]) -> str | None:
    unavailable = [t for t in requested if t not in grantable]
    if not unavailable:
        return None
    offer = ", ".join(sorted(grantable)) if grantable else "(none)"
    return (f"cannot grant {', '.join(unavailable)}: not available to you right now, or "
            f"reserved to you (planning, delegation, asking the user, writing memory, "
            f"cluster submission). What you may grant: {offer}. Leave `tools` empty for a "
            f"read-only exploration.")


def budget(ctx: Context | None, level: str, requested: list[str],
            time_budget_secs: int) -> tuple[bool, int]:
    """Whether the child is working, and its wall time.

    A child with a writing or executing tool is working (it also gets a worktree);
    anything else is reconnaissance. Decided before the child connects any server.
    """
    working = level_allows(level, "build") and bool(set(requested) & writers(ctx))
    ceiling = (spawn_limits.SUBAGENT_WORKING_BUDGET_SECS if working
               else spawn_limits.SUBAGENT_EXPLORE_BUDGET_SECS)
    budget = max(spawn_limits.SUBAGENT_MIN_BUDGET_SECS,
                 min(int(time_budget_secs or ceiling), ceiling))
    return working, budget


def resolve_model(ctx: Context | None, model: str,
                   served: list[dict]) -> tuple[str, str | None]:
    """The model the child runs on (the caller's unless another is named), or a refusal.

    *served* is what the endpoint serves; a named model must be among them and fit to
    run a sub-agent.
    """
    own = model_of(ctx)
    model = (model or "").strip()
    if model and model != own:
        from .model_catalog import subagent_model_refusal
        return model, subagent_model_refusal(model, served)
    return own, None

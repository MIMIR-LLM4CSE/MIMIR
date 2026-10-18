"""Which guidance nudges may speak at each enforcement level and mode."""
from __future__ import annotations


# Guidance nudges permitted per (enforcement, mode), by cascade category name. Each
# nudge's own gate still applies; "off" is absent (the layer is skipped before this).
ALL_GUIDANCE = frozenset({
    "env_resolution", "doc",
    "blast_radius", "creation", "todo", "env_cleanup", "unwatched_run",
})


GUIDANCE_BY_LEVEL_MODE: dict[tuple[str, str], frozenset[str]] = {
    # strict babysits everything (the branches still gate mode themselves, so this does
    # not actually leak agent-only nudges into plan mode).
    ("strict", "agent"): ALL_GUIDANCE,
    ("strict", "plan"): ALL_GUIDANCE,
    # light: only the nudges that guard a costly, hard-to-detect mistake nothing later
    # corrects: blast_radius (breaking callers), env_cleanup (leftover env side
    # effects), unwatched_run (an overnight job nobody watches). Everything else is
    # hand-holding a capable model does unprompted. Validation is not guidance at all:
    # it is a verification nudge, on at every level.
    ("light", "agent"): frozenset({"blast_radius", "env_cleanup", "unwatched_run"}),
    ("light", "plan"): frozenset(),
    # ask is answer-only: nothing is planned and nothing is edited, so no guidance
    # category has anything to guard. Listed explicitly rather than relying on the
    # missing-key default so the table stays the readable source of truth.
    ("strict", "ask"): frozenset(),
    ("light", "ask"): frozenset(),
}


def guidance_enabled(category: str, *, enforcement: str, active_mode: str) -> bool:
    """True if *category* may fire at this enforcement level and mode (table lookup).

    Governs ONLY the enforcement dimension — the caller still applies each nudge's
    own ``active_mode``/situational conditions. See ``GUIDANCE_BY_LEVEL_MODE``.
    """
    return category in GUIDANCE_BY_LEVEL_MODE.get((enforcement, active_mode), frozenset())

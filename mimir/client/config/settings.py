"""The vocabulary of the user's settings: what each one may be, and how it is read.

Thinking depth, how far a sub-agent may go, sampling temperature, the closed sets of
the other settings. Values and parsing only — applying a setting to an agent is
``core.settings``'s business.
"""
from __future__ import annotations

CONTEXT_MODES: tuple[str, ...] = ("compact", "full")
ENFORCEMENT_LEVELS: tuple[str, ...] = ("strict", "light", "off")

# Who answers the approval cards: ``manual`` (the user), ``auto`` (sensitive tools
# pass, leaving the workspace still asks), ``auto_all`` (nothing asks).
APPROVAL_MODES: tuple[str, ...] = ("manual", "auto", "auto_all")
# What the user types for ``auto_all``: nobody types an underscore at a prompt.
APPROVAL_MODE_ALIASES: dict[str, str] = {"all": "auto_all", "auto-all": "auto_all"}

BACKENDS: tuple[str, ...] = ("ollama", "vllm", "ray")

# Thinking-depth scale — the single source of truth for the reasoning-depth ladder
# exposed by the CLI (/think) and the WS command (/thinking-depth), and mirrored by
# the webview slider. Index = depth level.
#
# ``auto`` is the default: thinking is on with no imposed budget, and the system
# prompt carries a calibration directive telling the model to close the reasoning
# block immediately on trivial turns and spend a long chain only where the task is
# genuinely uncertain. The fixed rungs (quick/medium/deep) impose a token budget
# instead, which the per-phase scaling in the agent loop then modulates; ``max``
# is thinking-on, unbudgeted, without the calibration directive.
THINKING_DEPTH_LABELS: tuple[str, ...] = ("off", "auto", "quick", "medium", "deep", "max")
THINKING_DEPTH_BUDGETS: tuple[int, ...] = (-1, -1, 512, 4096, 16384, -1)
THINKING_DEPTH_AUTO: int = 1
DEFAULT_THINKING_DEPTH: int = THINKING_DEPTH_AUTO


def clamp_thinking_depth(level: int) -> int:
    """Clamp a depth level into the valid range of the ladder."""
    return max(0, min(len(THINKING_DEPTH_LABELS) - 1, int(level)))


def thinking_depth_from_label(label: str) -> int | None:
    """Resolve a depth label (``"auto"``, ``"deep"``, …) to its level, else None.

    ``on`` is accepted as an alias for ``auto``: ``/think on`` lets the model calibrate.
    """
    key = label.strip().lower()
    if key in ("on", "true", "1", "yes"):
        return THINKING_DEPTH_AUTO
    if key in ("false", "0", "no"):
        return 0
    try:
        return THINKING_DEPTH_LABELS.index(key)
    except ValueError:
        return None


# ── How much a sub-agent may do ───────────────────────────────────────────────
# A ladder, weakest first, each rung adding to the one before it. It is the user's
# call, not the model's: a child that writes, and more so one that opens a branch,
# commits something on their behalf.
#
#   explore   read, search, navigate; also what an unknown rung falls back to.
#   build     + editing and running commands — always in a git worktree of its own,
#             on its own branch. There is deliberately no rung in between: sub-agents
#             editing one shared tree overwrite each other in silence (one reads a
#             file, another rewrites it, the first writes back what it read), and no
#             arbitration the model can be trusted with fixes that. A sub-agent that
#             writes gets a copy, or it does not write.
SUBAGENT_LEVELS: tuple[str, ...] = ("explore", "build")
DEFAULT_SUBAGENT_LEVEL: str = SUBAGENT_LEVELS[0]

# Other names a rung is read under. "parallel" is in the preferences of everyone who set
# it, and falling back to "explore" would take a granted right away.
SUBAGENT_LEVEL_ALIASES: dict[str, str] = {"parallel": "build"}


def clamp_subagent_level(level: str) -> str:
    """Resolve a rung name, falling back to the one that concedes nothing."""
    name = (level or "").strip().lower()
    name = SUBAGENT_LEVEL_ALIASES.get(name, name)
    return name if name in SUBAGENT_LEVELS else DEFAULT_SUBAGENT_LEVEL


def subagent_level_allows(level: str, required: str) -> bool:
    """True when *level* is at least *required* on the ladder."""
    return SUBAGENT_LEVELS.index(clamp_subagent_level(level)) >= \
        SUBAGENT_LEVELS.index(clamp_subagent_level(required))


# Sampling temperature the user may set per model (/temperature, the ⚙ panel). None
# is "the model's own": nothing is sent and the server applies the model's
# generation_config. Only the backends below forward it; Anthropic's extended thinking
# requires temperature 1, so the setting is not offered there.
TEMPERATURE_MIN: float = 0.0
TEMPERATURE_MAX: float = 2.0
TEMPERATURE_BACKENDS: frozenset[str] = frozenset({"vllm", "ray", "ollama"})


def parse_temperature(text: str) -> tuple[bool, float | None]:
    """Parse a /temperature argument: ``(ok, value)``, value None meaning "default"."""
    key = (text or "").strip().lower()
    if key in ("default", "auto", "none", "model"):
        return True, None
    try:
        value = float(key)
    except ValueError:
        return False, None
    if not TEMPERATURE_MIN <= value <= TEMPERATURE_MAX:
        return False, None
    return True, value

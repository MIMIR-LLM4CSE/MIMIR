"""Session commands, one grammar for every front-end — see ``base``."""
from __future__ import annotations

from .agent import AGENT_COMMANDS
from .base import (
    Command,
    CommandHost,
    CommandRegistry,
    CommandResult,
    Item,
    apply_setting_or_error,
)
from .housekeeping import HOUSEKEEPING_COMMANDS
from .settings import SETTING_COMMANDS

# Every command a front-end may offer; each takes the ones for its surface.
COMMANDS = CommandRegistry(SETTING_COMMANDS + HOUSEKEEPING_COMMANDS + AGENT_COMMANDS)

__all__ = [
    "COMMANDS",
    "Command",
    "CommandHost",
    "CommandRegistry",
    "CommandResult",
    "Item",
    "apply_setting_or_error",
]

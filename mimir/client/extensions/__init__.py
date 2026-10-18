"""The user's extensions under ``.mimir/``: ``servers/``, ``skills/``, ``plugins/``.

Each location is env-overridable (``MIMIR_*_DIR``) and fail-open. A plugin pack imports
``PolicyCheck``, ``NudgeRule``, ``PostToolRule`` and their ``register_*`` from here: a
policy blocks before a call, a hook reacts after it, a nudge speaks at the end of a turn.
"""

from __future__ import annotations

# Plugin-pack authoring surface (each descriptor lives next to its consumer).
from ..core.extension_set import ExtensionSet, registering_into
from ..guardrails.policy.plugins import PolicyCheck, register_policy_check
from ..guardrails.nudges.plugins import NudgeRule, register_nudge
from ..tool_execution.plugins import PostToolRule, register_post_tool

# .mimir/ discovery + loading, one module per extension type.
from .servers import (
    all_servers,
    all_server_descriptions,
    discover_user_servers,
    resolve_servers_dir,
)
from .skills import resolve_skills_dir
from .system_prompt import resolve_system_prompt_file
from .plugins import load_plugins, resolve_plugins_dir

__all__ = [
    # authoring surface
    "ExtensionSet",
    "registering_into",
    "PolicyCheck",
    "register_policy_check",
    "NudgeRule",
    "register_nudge",
    "PostToolRule",
    "register_post_tool",
    # servers
    "all_servers",
    "all_server_descriptions",
    "discover_user_servers",
    "resolve_servers_dir",
    # skills
    "resolve_skills_dir",
    # system prompt override
    "resolve_system_prompt_file",
    # plugins
    "load_plugins",
    "resolve_plugins_dir",
]

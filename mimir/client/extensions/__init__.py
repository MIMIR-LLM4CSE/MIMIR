"""User extensions under the workspace ``.mimir/`` — the single home for everything
the user drops in: MCP **servers**, **skills**, and policy/nudge **plugin packs**.

  ``.mimir/servers/server_<name>.py|.js``  (override ``MIMIR_SERVERS_DIR``)  → servers.py
  ``.mimir/skills/<name>/SKILL.md``        (override ``MIMIR_SKILLS_DIR``)   → skills.py
  ``.mimir/plugins/*.py``                  (override ``MIMIR_PLUGINS_DIR``)  → plugins.py

Each is env-overridable and fail-open. The raw path primitives (``MIMIR_DIR``,
``resolve_extension_dir``, the ``*_DIRNAME``/``*_DIR_ENV`` constants) live in
``config.constants``; this package owns discovery + loading.

Public authoring surface for a plugin pack (dropped into the plugins dir)::

    from mimir.client.extensions import (
        PolicyCheck, NudgeRule, register_policy_check, register_nudge,
    )

Policies are locked (mandatory); nudges are toggleable. Reference *capabilities* from
``mimir.client.context.capabilities``, never literal tool names.
"""

from __future__ import annotations

# Plugin-pack authoring surface (descriptors + registries live in guardrails/).
from ..guardrails.policy.plugins import PolicyCheck, PolicyRegistry, register_policy_check
from ..guardrails.nudges.plugins import NudgeRule, NudgeRegistry, register_nudge

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
    "PolicyCheck",
    "PolicyRegistry",
    "register_policy_check",
    "NudgeRule",
    "NudgeRegistry",
    "register_nudge",
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

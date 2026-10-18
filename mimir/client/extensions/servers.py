"""Discover the user's MCP servers (``.mimir/servers/server_<name>.py|.js``).

Merged with the bundled ``SERVERS``; a name clashing with a bundled one is skipped.
Import-light, so any layer can use it.
"""

from __future__ import annotations

import logging
import os

from ..config.servers import SERVER_DESCRIPTIONS, SERVERS
from ..config.paths import SERVERS_DIR_ENV, SERVERS_DIRNAME, resolve_extension_dir

logger = logging.getLogger(__name__)


def resolve_servers_dir() -> str:
    """User servers directory: ``MIMIR_SERVERS_DIR`` env, else ``.mimir/servers``."""
    return resolve_extension_dir(SERVERS_DIR_ENV, SERVERS_DIRNAME)


def _server_name_from_filename(filename: str) -> str:
    """``server_weather.py`` → ``weather``; ``weather.js`` → ``weather``."""
    stem = os.path.splitext(filename)[0]
    if stem.startswith("server_"):
        stem = stem[len("server_"):]
    return stem


def discover_user_servers(servers_dir: str | None = None) -> dict[str, str]:
    """Return ``{name: abspath}`` for user MCP servers under the servers dir.

    Scans ``*.py``/``*.js`` (skips ``_``-prefixed). A discovered name that collides
    with a bundled server is skipped with a warning (the core is protected).
    """
    directory = servers_dir or resolve_servers_dir()
    if not os.path.isdir(directory):
        return {}
    found: dict[str, str] = {}
    for entry in sorted(os.listdir(directory)):
        if entry.startswith("_") or not entry.endswith((".py", ".js")):
            continue
        name = _server_name_from_filename(entry)
        if not name:
            continue
        if name in SERVERS:
            logger.warning(
                "Ignoring user server '%s' (%s): name collides with a bundled server.",
                name, entry,
            )
            continue
        if name in found:
            logger.warning("Duplicate user server name '%s'; keeping first.", name)
            continue
        found[name] = os.path.join(directory, entry)
    return found


def all_servers() -> dict[str, str]:
    """Merged connect registry: bundled ``SERVERS`` + discovered user servers.

    Bundled entries come first and cannot be shadowed (collisions were already
    filtered out in ``discover_user_servers``).
    """
    return {**SERVERS, **discover_user_servers()}


def all_server_descriptions() -> dict[str, str]:
    """Bundled ``SERVER_DESCRIPTIONS`` plus a generic line per discovered user server."""
    merged = dict(SERVER_DESCRIPTIONS)
    for name, path in discover_user_servers().items():
        merged.setdefault(name, f"Custom server (.mimir/servers): {os.path.basename(path)}")
    return merged

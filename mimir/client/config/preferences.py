"""Operator preferences persisted per-workspace in the central state dir at
``<STATE_DIR>/preferences.json`` (see config.constants.STATE_DIR).

This is *operator config* — which MCP servers and skills the user has switched off
from the toggle panel — part of the agent STATE (alongside the agent's own memory
under ``<STATE_DIR>/memory/``), kept out of the workspace. The client reads it at
startup to decide which servers' tools
to advertise to the LLM and which skills are eligible for auto-detection; the LLM
never reads this file.

Schema::

    {
      "disabled_servers": ["strings", "datetime"],
      "disabled_skills":  ["proxy-optimize"],
      "disabled_nudges":  ["authz_reminder"]
    }

Only *disabled* names are stored (an absent name is enabled), so newly added servers,
skills, and application nudges default to on without needing a migration.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .constants import STATE_DIR

logger = logging.getLogger(__name__)

PREFERENCES_FILENAME = "preferences.json"


def _preferences_path() -> str:
    return os.path.join(STATE_DIR, PREFERENCES_FILENAME)


def load_preferences() -> dict[str, Any]:
    """Return the parsed preferences dict, or an empty one if absent/unreadable."""
    path = _preferences_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except FileNotFoundError:
        return {}
    except Exception as exc:  # corrupt file — don't crash startup
        logger.warning("Could not read %s: %s", path, exc)
    return {}


def load_disabled() -> tuple[set[str], set[str], set[str]]:
    """Return ``(disabled_servers, disabled_skills, disabled_nudges)`` as sets."""
    data = load_preferences()

    def _as_set(key: str) -> set[str]:
        value = data.get(key, [])
        return {str(x) for x in value} if isinstance(value, list) else set()

    return _as_set("disabled_servers"), _as_set("disabled_skills"), _as_set("disabled_nudges")


def save_disabled(
    disabled_servers: set[str],
    disabled_skills: set[str],
    disabled_nudges: set[str] | None = None,
) -> None:
    """Persist the disabled sets to ``.mimir/preferences.json`` (sorted, atomic)."""
    path = _preferences_path()
    payload = {
        "disabled_servers": sorted(disabled_servers),
        "disabled_skills": sorted(disabled_skills),
        "disabled_nudges": sorted(disabled_nudges or set()),
    }
    try:
        os.makedirs(STATE_DIR, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Could not write %s: %s", path, exc)

"""Resolution of the user system-prompt override under ``.mimir/system_prompt.md``.

A user file ``.mimir/system_prompt.md`` (override ``MIMIR_SYSTEM_PROMPT_FILE``, or an
explicit path) *replaces* the agent's built-in base system prompt. This module only
resolves *which* file to use, the same drop-in / env-overridable convention as the
servers / skills / plugins resolvers; the reading + assembly lives in
``prompt/system_prompt.py`` (``build_base_system_content``).
"""

from __future__ import annotations

import os

from ..config.constants import MIMIR_DIR, SYSTEM_PROMPT_ENV, SYSTEM_PROMPT_FILENAME


def resolve_system_prompt_file(override: str = "") -> str:
    """Return the path of the system-prompt override .md, or "" for the built-in default.

    Resolution order (first existing file wins):
      1. an explicit ``override`` path (highest precedence),
      2. the ``MIMIR_SYSTEM_PROMPT_FILE`` env var,
      3. ``<workspace>/.mimir/system_prompt.md``.
    """
    candidates = [
        override,
        os.environ.get(SYSTEM_PROMPT_ENV, "").strip(),
        os.path.join(MIMIR_DIR, SYSTEM_PROMPT_FILENAME),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""

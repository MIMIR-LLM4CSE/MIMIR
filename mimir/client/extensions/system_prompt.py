"""Which file overrides the doctrine half of the base prompt (``.mimir/system_prompt.md``).

The core half is appended either way; assembly is in ``prompt/system_prompt.py``.
"""

from __future__ import annotations

import os

from ..config.paths import MIMIR_DIR, SYSTEM_PROMPT_ENV, SYSTEM_PROMPT_FILENAME


def resolve_system_prompt_file(override: str = "") -> str:
    """The override file: *override*, else ``MIMIR_SYSTEM_PROMPT_FILE``, else ``.mimir/system_prompt.md``; "" if none."""
    candidates = [
        override,
        os.environ.get(SYSTEM_PROMPT_ENV, "").strip(),
        os.path.join(MIMIR_DIR, SYSTEM_PROMPT_FILENAME),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return ""

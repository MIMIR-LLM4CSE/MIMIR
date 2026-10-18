"""The session allowlist of out-of-workspace paths the user approved.

The client writes ``<session_state_dir()>/approved_paths.json``; the sandbox guards read
it per call (the only live client→server channel) as ``extra_roots``. One file per
session, since consent is given per conversation. A missing or corrupt file allows nothing.
"""

import json
import os

from .state_paths import session_state_dir

_APPROVED_FILE = "approved_paths.json"


def approved_paths_file() -> str:
    return os.path.join(session_state_dir(), _APPROVED_FILE)


def approved_roots() -> list[str]:
    """Absolute paths the user approved this session, or ``[]`` (fail-closed)."""
    path = approved_paths_file()
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [str(p) for p in data if isinstance(p, str) and p.strip()]

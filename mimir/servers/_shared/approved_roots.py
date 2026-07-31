"""Session allowlist of out-of-workspace paths the user explicitly approved.

The client is the sole writer: when the user approves an out-of-workspace access
(allow-once or always) it appends the absolute path to
``<state_dir()>/approved_paths.json`` and resets the file at session start. The
sandbox guards read it **per call** (the servers' env is frozen at spawn, so a
file on the shared state dir is the only live client→server channel) and pass the
entries as ``extra_roots`` to ``resolve_path_in_root``.

Read-only + best-effort: any missing/corrupt file yields ``[]`` (fail-closed —
nothing extra is allowed), so a broken allowlist can never widen the sandbox.
"""

import json
import os

from state_paths import state_dir

_APPROVED_FILE = "approved_paths.json"


def approved_paths_file() -> str:
    return os.path.join(state_dir(), _APPROVED_FILE)


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

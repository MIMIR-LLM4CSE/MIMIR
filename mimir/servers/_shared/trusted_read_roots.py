"""The out-of-workspace roots read without approval: the state dir, where MIMIR keeps its
run artefacts. Shared by the servers and the client gate; stdlib only.
"""

import os


def trusted_read_roots() -> list[str]:
    """The central state dir (``MIMIR_STATE_DIR``) when set, unresolved.

    The client adds its own ``STATE_DIR``, which its process has no variable for.
    """
    state_dir = os.environ.get("MIMIR_STATE_DIR")
    return [state_dir] if state_dir else []

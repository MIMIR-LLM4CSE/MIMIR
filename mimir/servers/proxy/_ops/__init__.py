"""Op implementations for the proxy MCP server.

Each module holds plain functions returning ``ok()``/``err()`` dicts — no
``@mcp.tool`` decorators here.  The seven dispatch tools live in
``server_proxy.py`` and route to these functions by ``op`` name.
"""

import os
import sys

_PROXY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_PROXY_DIR, "..", "_shared"), _PROXY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from responses import err, ok  # noqa: E402,F401


def _with_next(payload: dict, next_step: str) -> dict:
    """Attach the sequencing hint naming the exact next call to make."""
    payload["next_step"] = next_step
    return payload

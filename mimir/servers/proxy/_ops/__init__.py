"""Op implementations for the proxy MCP server.

Each module holds plain functions returning ``ok()``/``err()`` dicts — no
``@mcp.tool`` decorators here.  The seven dispatch tools live in
``server_proxy.py`` and route to these functions by ``op`` name.
"""

import os
import re

from ..._shared.responses import err, ok  # noqa: F401

# The proxy server's directory: where a generated proxy script finds its helpers.
_PROXY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _with_next(payload: dict, next_step: str) -> dict:
    """Attach the sequencing hint naming the exact next call to make."""
    payload["next_step"] = next_step
    return payload


def _check_name(label: str, value: str):
    """Reject names that would escape their store directory; None when valid."""
    if not value or not re.match(r"^[A-Za-z0-9_\-]+$", value):
        return err(f"{label} must be non-empty and contain only [A-Za-z0-9_-].")
    return None

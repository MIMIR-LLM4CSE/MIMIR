"""Canonical trusted out-of-workspace read roots (single source of truth).

Agent-produced run artefacts — proxy/HPC job logs, central session state — live
outside the workspace but are always safe to read. Both the read-only workspace
servers (which widen their sandbox to these) and the client policy gate (which
skips the approval prompt for reads under them) must agree on this set, so it is
defined here once. Dependency-free (stdlib only) so it imports cleanly on either
side of the client/server process boundary — flat ``from trusted_read_roots import
...`` in a server subprocess, packaged ``mimir.servers._shared.trusted_read_roots``
from the client (cf. ``embed.py``).
"""

import os

# Fixed cache locations for agent run artefacts (no env dependency).
TRUSTED_CACHE_ROOTS = (
    "~/.cache/proxy_bench",   # proxy runs + optimization sessions
    "~/.cache/mimir_hpc",     # HPC batch job dirs
)


def trusted_read_roots() -> list[str]:
    """The fixed caches plus the central state dir (``MIMIR_STATE_DIR`` if set).

    Not realpath-resolved here — callers normalize as needed (the client
    realpaths for its ``startswith`` containment check; the servers pass these as
    ``extra_roots`` which resolves them itself).
    """
    roots = [os.path.expanduser(r) for r in TRUSTED_CACHE_ROOTS]
    state_dir = os.environ.get("MIMIR_STATE_DIR")
    if state_dir:
        roots.append(state_dir)
    return roots

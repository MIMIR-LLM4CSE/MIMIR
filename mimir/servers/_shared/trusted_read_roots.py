"""Canonical trusted out-of-workspace read roots (single source of truth).

Agent-produced run artefacts — job scripts, Slurm and shell logs, session state —
live outside the workspace but are always safe to read. Both the read-only workspace
servers (which widen their sandbox to these) and the client policy gate (which
skips the approval prompt for reads under them) must agree on this set, so it is
defined here once. Dependency-free (stdlib only) so it imports cleanly on either
side of the client/server process boundary — flat ``from trusted_read_roots import
...`` in a server subprocess, packaged ``mimir.servers._shared.trusted_read_roots``
from the client (cf. ``embed.py``).
"""

import os


def trusted_read_roots() -> list[str]:
    """The central state dir (``MIMIR_STATE_DIR``), when one is set.

    Not realpath-resolved here — callers normalize as needed (the client
    realpaths for its ``startswith`` containment check; the servers pass these as
    ``extra_roots`` which resolves them itself).

    The client appends its own ``STATE_DIR`` where it needs to, because
    ``MIMIR_STATE_DIR`` is placed only in the server subprocesses' environment — see
    guardrails/policy/gates.py and tool_execution/validation.py.
    """
    state_dir = os.environ.get("MIMIR_STATE_DIR")
    return [state_dir] if state_dir else []

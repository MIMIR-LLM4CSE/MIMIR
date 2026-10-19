"""Where things are: the bundled servers and skills, the workspace, the state dirs.

Resolved once, at import, from the environment — one client process keeps one answer
for the whole run.
"""
from __future__ import annotations

import os

# The workspace id names both the state dir below and the scratchpad under /tmp, so
# it is defined once, server-side, and imported here rather than duplicated. The
# .mimir extension paths come from server-side too: the mimir_api server reports the
# drop-in locations this module's loaders scan, so both ends read one definition.
from ...shared.state_paths import (  # noqa: F401  (re-exported)
    global_state_dir,
    workspace_id,
)
from ...shared.extension_paths import (  # noqa: F401  (re-exported)
    MIMIR_DIRNAME,
    PLUGINS_DIR_ENV,
    PLUGINS_DIRNAME,
    SERVERS_DIR_ENV,
    SERVERS_DIRNAME,
    SKILLS_DIR_ENV,
    SKILLS_DIRNAME,
    SYSTEM_PROMPT_ENV,
    SYSTEM_PROMPT_FILENAME,
    mimir_dir as _mimir_dir,
    resolve_extension_dir as _resolve_extension_dir,
    workspace_root as _workspace_root,
)


SERVER_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "servers")) + os.sep
SKILL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "skills")) + os.sep

# Workspace root: the folder the agent operates on. Matches the convention the
# MCP servers use (MCP_FILES_ROOT, set by server_manager / the extension), so the
# client and the server subprocesses resolve .mimir to the same place. Falls back
# to the process cwd when MCP_FILES_ROOT is unset.
WORKSPACE_ROOT = _workspace_root()

# Per-workspace *extensions* directory (<workspace>/.mimir). Holds ONLY the user
# extensions that belong with the repo: plugins/, skills/, servers/, and the
# system_prompt.md template. The agent's own STATE (memory, sessions, todos,
# plans, preferences.json, active_session) is in STATE_DIR below. Resolved the same way the UI front-ends and servers
# resolve it, so the extension path stays in agreement.
MIMIR_DIR = _mimir_dir(WORKSPACE_ROOT)

# Per-workspace state, outside the workspace: <STATE_HOME>/<basename-hash>/.
# MIMIR_STATE_DIR overrides it whole, MIMIR_STATE_HOME its root (~/.mimir). Servers get
# the same directory through MIMIR_STATE_DIR.
STATE_HOME = os.path.abspath(
    os.environ.get("MIMIR_STATE_HOME")
    or os.path.join(os.path.expanduser("~"), ".mimir")
)


STATE_DIR = os.path.abspath(
    os.environ.get("MIMIR_STATE_DIR")
    or os.path.join(STATE_HOME, workspace_id(WORKSPACE_ROOT))
)

# The third tier: state shared by EVERY workspace, <STATE_HOME>/global/. Resolved here
# rather than left to the servers because the client is the end that knows STATE_HOME —
# MIMIR_STATE_DIR goes only into the server subprocesses' environment, never this one —
# and the resolved path is published to them as MIMIR_GLOBAL_STATE_DIR (server_manager),
# so both ends agree without either re-deriving it. Holds the global memory store.
GLOBAL_STATE_DIR = global_state_dir(STATE_HOME)

# The four extension locations (env var if set, else MIMIR_DIR/<name>; see
# shared/extension_paths.py): the base system prompt, plugins (policy, post-tool
# and nudge packs), skills (merged over the bundled ones) and servers (a name clashing
# with a bundled one is skipped).


def resolve_extension_dir(env_var: str, dirname: str) -> str:
    """Resolve a user-extension directory: ``$env_var`` if set, else ``MIMIR_DIR/dirname``.

    Pinned to the ``MIMIR_DIR`` this module resolved at import, so one client process
    keeps one answer for the whole run.
    """
    return _resolve_extension_dir(env_var, dirname, MIMIR_DIR)


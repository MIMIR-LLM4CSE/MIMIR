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
from ...servers._shared.state_paths import (  # noqa: F401  (re-exported)
    global_state_dir,
    workspace_id,
)
from ...servers._shared.extension_paths import (  # noqa: F401  (re-exported)
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
# plans, preferences.json, active_session) no longer lives
# here — see STATE_DIR below. Resolved the same way the UI front-ends and servers
# resolve it, so the extension path stays in agreement.
MIMIR_DIR = _mimir_dir(WORKSPACE_ROOT)

# Central per-workspace STATE directory, kept OUT of the workspace so the agent's
# runtime state never pollutes the repo. Layout: <STATE_HOME>/<workspace-id>/,
# where the id is a readable basename plus a short hash of the absolute workspace
# path (so two different checkouts that share a basename don't collide). Overrides:
#   MIMIR_STATE_DIR  — full path, wins over everything (tests / headless runs)
#   MIMIR_STATE_HOME — the root under which per-workspace dirs are created (~/.mimir)
# The same STATE_DIR is handed to the server subprocesses via MIMIR_STATE_DIR
# (server_manager), so client and servers resolve state to the same place.
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

# The four extension locations re-exported at the top of this module (their env vars
# and default names live in servers/_shared/extension_paths.py). Resolution is always
# "env var if set, else MIMIR_DIR/<name>", through resolve_extension_dir below:
#   SYSTEM_PROMPT_ENV / SYSTEM_PROMPT_FILENAME — the base prompt, a single file whose
#     env form needs no .mimir/ dir at all (discovery/context_builder.py). Nothing is
#     auto-created in the workspace; a template lives in mimir/examples/system_prompt.md.
#   PLUGINS_DIR_ENV / PLUGINS_DIRNAME — policy, post-tool and nudge packs; each *.py in
#     the dir is imported once at agent init and registers on import (extensions/plugins.py).
#   SKILLS_DIR_ENV / SKILLS_DIRNAME — user skills, merged over the bundled set, same name
#     overriding (agent_core.load_skills).
#   SERVERS_DIR_ENV / SERVERS_DIRNAME — user MCP servers; a name colliding with a bundled
#     one is skipped, so the core cannot be shadowed (extensions/servers.py).


def resolve_extension_dir(env_var: str, dirname: str) -> str:
    """Resolve a user-extension directory: ``$env_var`` if set, else ``MIMIR_DIR/dirname``.

    Pinned to the ``MIMIR_DIR`` this module resolved at import, so one client process
    keeps one answer for the whole run.
    """
    return _resolve_extension_dir(env_var, dirname, MIMIR_DIR)


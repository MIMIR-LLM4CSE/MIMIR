"""Single source of truth for the user-extension directories under ``.mimir/``.

The client *loads* these directories (skills, servers, plugin packs, the base prompt)
and the ``mimir_api`` server *reports* them, so the names and the resolution order are
needed on both sides of the client/server process boundary. They live here, the way
``workspace_id`` does in ``state_paths``, and ``client/config/constants`` re-exports
them rather than restating them — a drop-in path a user is told about must be the one
the loader actually scans.

Dependency-free (stdlib only, no sibling imports) so it loads both ways: flat
``from extension_paths import ...`` in a server subprocess, packaged
``mimir.servers._shared.extension_paths`` from the client.
"""

import os

# The extensions directory itself, inside the workspace. Holds only what belongs with
# the repo; the agent's runtime state lives elsewhere (see state_paths).
MIMIR_DIRNAME = ".mimir"

# Per extension type: the env var that overrides its location, and its default
# directory name under .mimir/. The base prompt is a single file, hence a filename.
PLUGINS_DIR_ENV = "MIMIR_PLUGINS_DIR"
PLUGINS_DIRNAME = "plugins"
SKILLS_DIR_ENV = "MIMIR_SKILLS_DIR"
SKILLS_DIRNAME = "skills"
SERVERS_DIR_ENV = "MIMIR_SERVERS_DIR"
SERVERS_DIRNAME = "servers"
SYSTEM_PROMPT_ENV = "MIMIR_SYSTEM_PROMPT_FILE"
SYSTEM_PROMPT_FILENAME = "system_prompt.md"


def workspace_root() -> str:
    """The folder the agent operates on: ``MCP_FILES_ROOT``, else the process cwd.

    The client resolves this once at startup and puts it in the server subprocesses'
    environment, so both ends resolve ``.mimir`` to the same place.
    """
    return os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())


def mimir_dir(root: str | None = None) -> str:
    """The workspace extensions directory, ``<workspace>/.mimir``."""
    return os.path.join(root or workspace_root(), MIMIR_DIRNAME)


def resolve_extension_dir(env_var: str, dirname: str, base: str | None = None) -> str:
    """Resolve one extension directory: ``$env_var`` if set, else ``<base>/dirname``.

    *base* defaults to the live :func:`mimir_dir`; the client passes the one it
    resolved at import so a single process keeps one answer.
    """
    return os.path.abspath(os.environ.get(env_var) or os.path.join(base or mimir_dir(), dirname))

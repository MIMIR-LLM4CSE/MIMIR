"""How a bundled server or helper script is started: through ``servers/_launch.py``.

``launch_argv(module, *args)`` is the argv after the interpreter. Starting by module
through the launcher keeps the current directory (the user's workspace) off
``sys.path`` and gives the module its package, so its relative imports resolve.
"""
from __future__ import annotations

import os

_SERVERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAUNCHER = os.path.join(_SERVERS_DIR, "_launch.py")
# The directory holding the ``mimir`` package, for a generated script that imports it.
PACKAGE_PARENT = os.path.dirname(os.path.dirname(_SERVERS_DIR))


def launch_argv(module: str, *args: str) -> list[str]:
    return [LAUNCHER, module, *args]

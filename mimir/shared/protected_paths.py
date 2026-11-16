"""Workspace paths whose contents run later: a write there is asked about like leaving it.

``.mimir/`` holds the plugins and servers MIMIR loads at its next start, ``.git/`` the
configuration and hooks git runs (``core.fsmonitor`` fires on a plain ``git status``),
``.vscode/`` the tasks and settings the editor applies, and the proxy registry the
command ``proxy_exec`` runs. A file written there is code waiting for its moment, so
the approval that guards running code has to guard writing it.
"""
from __future__ import annotations

import os

# Directories protected wherever they sit in the workspace (a submodule has its own .git).
_PROTECTED_DIRS = frozenset({".git", ".mimir", ".vscode"})
# Files protected at their workspace-relative path.
_PROTECTED_FILES = frozenset({os.path.join("proxy_bench", "registry.json")})

# What MIMIR's own git calls in a workspace repository pass first: no fsmonitor, no
# hooks, whatever that repository's configuration says. A server commits and merges
# without anyone's approval, so it must not run what a planted .git/config names.
GIT_INERT = ("-c", "core.fsmonitor=", "-c", "core.hooksPath=/dev/null")

# How the card names why the call is held.
PROTECTED_TAG = "runs code later"
OUTSIDE_TAG = "outside workspace"


def is_protected(root: str, path: str) -> bool:
    """Whether absolute *path*, inside workspace *root*, is one whose contents run later."""
    rel = os.path.relpath(path, root)
    if rel == os.curdir or rel.startswith(os.pardir):
        return False
    parts = rel.split(os.sep)
    return bool(_PROTECTED_DIRS.intersection(parts)) or rel in _PROTECTED_FILES


def boundary_tag(root: str, paths: list[str]) -> str:
    """Why a call naming *paths* is held: they run code later, or they are outside."""
    root = os.path.realpath(root)
    inside = [p for p in paths if p == root or p.startswith(root + os.sep)]
    if inside and len(inside) == len(paths) and all(is_protected(root, p) for p in inside):
        return PROTECTED_TAG
    return OUTSIDE_TAG

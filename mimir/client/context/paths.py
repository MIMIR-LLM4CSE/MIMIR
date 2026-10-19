"""Pure path rules every layer shares: workspace-relative forms and the scratchpad."""
from __future__ import annotations

import os

from ...shared.state_paths import standing_roots
from ..config import paths


def absolute_workspace_path(path: str, *, cwd: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(cwd or os.getcwd(), path))


def normalize_workspace_path(path: str | None, *, cwd: str | None = None) -> str:
    """*path* relative to the workspace when it lies inside it, normalised either way."""
    if not path:
        return ""
    normalized = os.path.normpath(str(path).strip())
    if os.path.isabs(normalized):
        root = os.path.normpath(cwd or os.getcwd())
        if normalized == root or normalized.startswith(root + os.sep):
            return os.path.normpath(os.path.relpath(normalized, root))
    return normalized


def parent_path(path: str) -> str:
    return os.path.dirname(os.path.normpath(path)) or "."


def is_python_test_filepath(path: str) -> bool:
    basename = os.path.basename(path)
    return path.endswith(".py") and (basename.startswith("test_") or basename.endswith("_test.py"))


def scratch_roots() -> list[str]:
    """The agent's scratchpad directories, as absolute real paths.

    ``paths.STATE_DIR`` is passed because the active-session sidecar lives there, and
    ``MIMIR_STATE_DIR`` is set only in the servers' environment.
    """
    return [os.path.realpath(os.path.abspath(r)) for r in standing_roots(paths.STATE_DIR)]


def is_scratch_path(path: str) -> bool:
    """Whether *path* is in the scratchpad: working material, not a deliverable.

    Scratch files are not recorded as produced work, need no validation and stay out
    of the change ledger.
    """
    if not path:
        return False
    full = os.path.realpath(os.path.abspath(path))
    return any(full == root or full.startswith(root + os.sep) for root in scratch_roots())

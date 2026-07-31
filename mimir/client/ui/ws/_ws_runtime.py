"""Shared runtime foundation for the WebSocket-server modules.

Split out of ``ws_server.py`` so ``_AgentWorker`` (``ws_worker``) and ``_Session``
(``ws_session``) can share the stdout router, the session-scoped todo helpers, and
the context-budget constants without a circular import.

IMPORTANT — cwd bootstrap ordering: ``_bootstrap_cwd()`` MUST run before any import
that captures the workspace root *at import time* — ``config.constants`` resolves
WORKSPACE_ROOT / MIMIR_DIR from MCP_FILES_ROOT or os.getcwd() when first imported,
and the backend factory / ``resource_context`` do likewise. The ``--cwd`` flag is
only applied in ``main()``, which runs after module-level imports, so without this
early hook ``.mimir`` would be created against the launch directory. This module is
imported first by both ``ws_worker`` and ``ws_session``, so applying ``--cwd`` here
guarantees correct resolution regardless of which module is imported first.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable


# ── Working-directory bootstrap (MUST run before any cwd-dependent import) ──────
def _bootstrap_cwd() -> None:
    argv = sys.argv[1:]
    target: str | None = None
    for i, tok in enumerate(argv):
        if tok == "--cwd":
            target = argv[i + 1] if i + 1 < len(argv) else None
        elif tok.startswith("--cwd="):
            target = tok.split("=", 1)[1]
    if target:
        os.chdir(target)
        # Pin the sandbox/workspace root so constants and the MCP servers all
        # resolve `.mimir` (and MCP_FILES_ROOT) to the same place.
        os.environ.setdefault("MCP_FILES_ROOT", os.getcwd())


_bootstrap_cwd()

# Re-exported (imported *after* _bootstrap_cwd so the workspace root is captured
# correctly) for the ws modules to pull through this runtime — see module docstring.
from ...query_engine.backends.factory import get_backend
from ...context.resource_context import augment_query_with_resources


# ── Thread-aware stdout router ─────────────────────────────────────────────────

class _ThreadAwareOutput:
    """Routes sys.stdout writes to per-thread callbacks.

    Writes from unregistered threads fall back to the original stdout.
    Install once at server startup; individual agent threads register a sink.
    """

    def __init__(self, original: Any) -> None:
        self._original = original
        self._handlers: dict[int, Callable[[str], None]] = {}
        self._lock = threading.Lock()

    def register(self, thread_id: int, callback: Callable[[str], None]) -> None:
        with self._lock:
            self._handlers[thread_id] = callback

    def unregister(self, thread_id: int) -> None:
        with self._lock:
            self._handlers.pop(thread_id, None)

    def write(self, text: str) -> None:
        with self._lock:
            handler = self._handlers.get(threading.get_ident())
        if handler:
            handler(text)
        else:
            self._original.write(text)

    def flush(self) -> None:
        self._original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


_ORIGINAL_STDOUT = sys.stdout
_ROUTER = _ThreadAwareOutput(_ORIGINAL_STDOUT)
# Install the router once; uninstall on shutdown.
_ROUTER_INSTALLED = False

# ── Context-window budget constants ────────────────────────────────────────────
# Canonical values live in config.constants; aliased here to the private names
# this module already uses (kept in sync with chat_session.py via the shared source).
from ...config.constants import context_budget_for, CTX_TOTAL_FULL as _CTX_DEFAULT_FULL, STATE_DIR


def _ensure_router_installed() -> None:
    global _ROUTER_INSTALLED
    if not _ROUTER_INSTALLED:
        sys.stdout = _ROUTER
        _ROUTER_INSTALLED = True


# ── Session-scoped todo helpers (module-level so both classes can use them) ───

_MIMIR_DIR_WS = STATE_DIR


def _todo_file_for_session(session_id: str | None) -> str:
    """Return the session-scoped todo file path."""
    if session_id:
        return os.path.join(_MIMIR_DIR_WS, "sessions", session_id, "todo_list.md")
    return os.path.join(_MIMIR_DIR_WS, "todo_list.md")


def _write_active_session(session_id: str | None) -> None:
    """Write the active session ID to .mimir/active_session (sidecar)."""
    try:
        os.makedirs(_MIMIR_DIR_WS, exist_ok=True)
        sidecar = os.path.join(_MIMIR_DIR_WS, "active_session")
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(session_id or "")
    except Exception:
        pass


# Public surface pulled by the sibling ws modules. Listing the re-exports here
# (get_backend / augment_query_with_resources / context_budget_for / _CTX_DEFAULT_FULL)
# marks them intentional — they are imported through this runtime so its cwd
# bootstrap runs before the backend factory / config capture the workspace root.
__all__ = [
    "_ORIGINAL_STDOUT",
    "_ROUTER",
    "_ensure_router_installed",
    "_MIMIR_DIR_WS",
    "_todo_file_for_session",
    "_write_active_session",
    "get_backend",
    "augment_query_with_resources",
    "context_budget_for",
    "_CTX_DEFAULT_FULL",
]

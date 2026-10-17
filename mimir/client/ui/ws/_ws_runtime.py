"""Shared runtime foundation for the WebSocket-server modules.

Split out so ``_AgentWorker`` (``ws_worker``) and ``_Session`` (``ws_session``) can
share the stdout router, the session-scoped todo helpers and the context-budget
constants without a circular import. Importing it changes nothing: the router is
installed by ``serve``, and ``--cwd`` is applied by ``ws_server.main`` before any of
this is imported.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Callable

from ...config.limits import CTX_TOTAL_FULL as _CTX_DEFAULT_FULL
from ...config.paths import STATE_DIR
from ...context.resource_context import augment_query_with_resources
from ...query_engine.backends.factory import get_backend
from ...query_engine.context_budget import context_budget_for

logger = logging.getLogger(__name__)


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
        logger.debug("_write_active_session: a best-effort step failed and was skipped", exc_info=True)


# Public surface pulled by the sibling ws modules, re-exports included.
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

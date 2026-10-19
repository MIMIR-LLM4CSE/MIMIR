"""Which proxy this conversation is optimising, read off the proxy server's store.

Its own module because three guardrail modules ask (the policy gate, the evidence the
observations record, the completion report) and none of them owns the answer.
"""
from __future__ import annotations

import logging

from ...shared import proxy_store

logger = logging.getLogger(__name__)


def active_proxy_source(session_id: str | None = None) -> str:
    """The source file of the active proxy optimization session, or "" (best-effort).

    *session_id* names the asking conversation; without it the store's pointer answers,
    which only the evidence paths use (they never gate).
    """
    try:
        name = proxy_store.resolve_proxy_name("", session_id)
        return proxy_store.source_path(name) if name else ""
    except Exception:
        logger.debug("active_proxy_source: a best-effort step failed and was skipped", exc_info=True)
        return ""


def proxy_executable_paths(session_id: str | None = None) -> set[str]:
    """The source and the registered executable of this conversation's proxy, or empty.

    Not best-effort: the exec gate reads it, and an unreadable store must not open it.
    """
    name = proxy_store.resolve_proxy_name("", session_id)
    if not name:
        return set()
    return {p for p in (active_proxy_source(session_id), proxy_store.executable_path(name)) if p}

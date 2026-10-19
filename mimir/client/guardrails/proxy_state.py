"""Which proxy this conversation is optimising, read off the proxy server's own store.

Its own module because three guardrail modules ask (the policy gate, the evidence the
observations record, the completion report) and none of them owns the answer; nothing
else in the client reads the store. It is imported at the moment of asking: it is the
proxy server's library, heavy, and needed only by a conversation that optimises.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def active_proxy_source(session_id: str | None = None) -> str:
    """The source file of the active proxy optimization session, or "" (best-effort).

    *session_id* names the asking conversation; without it the store's pointer answers,
    which only the evidence paths use (they never gate).
    """
    try:
        from ...servers.proxy._lib import store

        name = store.resolve_proxy_name("", session_id)
        if not name:
            return ""
        cfg = store.read_json(store.opt_config_file(name))
        src = cfg.get("proxy_source_path") if isinstance(cfg, dict) else None
        return src if isinstance(src, str) else ""
    except Exception:
        logger.debug("active_proxy_source: a best-effort step failed and was skipped", exc_info=True)
        return ""


def proxy_executable_paths(session_id: str | None = None) -> set[str]:
    """The source and the registered executable of this conversation's proxy, or empty.

    Not best-effort: the exec gate reads it, and an unreadable store must not open it.
    """
    from ...servers.proxy._lib import store

    name = store.resolve_proxy_name("", session_id)
    if not name:
        return set()
    paths = set()
    src = active_proxy_source(session_id)
    if src:
        paths.add(src)
    reg = store.read_json(store.registry_path())
    entry = reg.get(name) if isinstance(reg, dict) else None
    exe = entry.get("executable_path") if isinstance(entry, dict) else None
    if isinstance(exe, str) and exe:
        paths.add(exe)
    return paths

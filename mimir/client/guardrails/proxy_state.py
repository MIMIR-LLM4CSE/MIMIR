"""Which proxy this conversation is optimising, read off the proxy server's own store.

Its own module because three guardrail modules ask (the policy gate, the evidence the
observations record, the completion report) and none of them owns the answer. The
store is imported at the moment of asking: it is the proxy server's library, heavy,
and needed only by a conversation that runs an optimisation at all.
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

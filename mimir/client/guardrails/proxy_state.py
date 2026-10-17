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
    """The source file of the currently active proxy optimization session, or "".

    Reads the proxy server's own store (single source of truth); an absent or
    unreadable session yields "" so every caller abstains. Best-effort by design —
    this informs guards and evidence, never correctness.

    *session_id* names the conversation asking. The store's pointer is per MIMIR
    session, and in this process N sessions share one environment, so a caller that
    knows which agent it is acting for must say. A caller that does not — the two
    evidence paths in guardrails/workflow.py and guardrails/observations.py, which
    receive only an execution_context — gets the pointer's own fallback, i.e. the
    session on screen. Imprecise under concurrency, and bounded: those two only ever
    raise a file's validation tier or add a reported line, never gate or block.
    """
    try:
        from ...servers.proxy._lib import store

        name = store._resolve_proxy_name("", session_id)
        if not name:
            return ""
        cfg = store._read_json(store._opt_config_file(name))
        src = cfg.get("proxy_source_path") if isinstance(cfg, dict) else None
        return src if isinstance(src, str) else ""
    except Exception:
        logger.debug("active_proxy_source: a best-effort step failed and was skipped", exc_info=True)
        return ""

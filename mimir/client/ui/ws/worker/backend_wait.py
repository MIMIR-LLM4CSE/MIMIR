"""Waiting for the model backend before the agent is built, and warming it up.

The wait reports its progress to the client, so a cold vLLM start shows what it is
doing rather than a frozen window."""
from __future__ import annotations

import asyncio
import logging
import os
from .....servers._shared.embed import verify_ssl
from ....llm.factory import get_backend
from ..runtime import (
    ORIGINAL_STDOUT,
)
from .outbox import Outbox

logger = logging.getLogger(__name__)


async def wait_for_backend(out_q: Outbox) -> None:
    """Poll the LLM backend until it answers, or fail fast when it never will.

    Asks the backend's own ``readiness_urls`` with its client and TLS policy, the second
    URL a fallback. A refused certificate or a 404 stops at once. Progress goes to the
    client and stdout. ``MIMIR_BACKEND_TIMEOUT`` (600 s), ``MIMIR_BACKEND_POLL_INTERVAL`` (5 s).
    """
    import httpx


    # Which endpoint, and what to ask it, is the backend's to say: a hosted API has
    # nothing to wait on, a self-hosted server names what to poll.
    backend = os.environ.get("LLM_BACKEND", "vllm")
    urls = get_backend().readiness_urls()
    if not urls:
        return
    health_url, fallback_url = urls[0], (urls[1] if len(urls) > 1 else "")

    timeout = int(os.environ.get("MIMIR_BACKEND_TIMEOUT", "600"))
    poll = float(os.environ.get("MIMIR_BACKEND_POLL_INTERVAL", "5"))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_progress = loop.time()
    progress_interval = 10.0

    def _say(text: str) -> None:
        out_q.put({"type": "output", "text": text})
        print(text.rstrip("\n"), file=ORIGINAL_STDOUT, flush=True)

    def _get(url: str) -> tuple[bool, str]:
        """(ready, permanent failure reason) for one GET. Empty reason: retry."""
        try:
            # trust_env=False: a corporate proxy has no route to the cluster and
            # swallows the request instead of refusing it. verify_ssl(): the same
            # switch the chat and embedding clients read, so the probe trusts
            # exactly what they trust.
            with httpx.Client(trust_env=False, timeout=4.0, verify=verify_ssl()) as client:
                resp = client.get(url)
        except httpx.ConnectError as exc:
            # A refused certificate is the one connection failure that will not
            # resolve itself: waiting cannot add a CA to the trust store.
            if "CERTIFICATE_VERIFY_FAILED" in str(exc) or "SSLCertVerificationError" in str(exc):
                return False, (
                    f"the server at {url} answered, but its security certificate "
                    f"is not trusted (common for internal servers). If you trust "
                    f"this server, untick “Vllm Verify Ssl” in the MIMIR settings "
                    f"(or set VLLM_VERIFY_SSL=0), then retry. Details: {exc}"
                )
            return False, ""
        except Exception:
            logger.debug("_get: a best-effort step failed and was skipped", exc_info=True)
            return False, ""
        if resp.status_code == 200:
            return True, ""
        # Up, but guarding the endpoint: the agent's own request carries the key.
        if resp.status_code in (401, 403):
            return True, ""
        if resp.status_code == 404:
            return False, f"404 — nothing is served at {url}"
        return False, ""

    def _check() -> tuple[bool, str]:
        ready, reason = _get(health_url)
        if ready or not fallback_url:
            return ready, reason
        # The engine may still be loading, when /health answers and /v1/models
        # does not. A fallback that answers outranks the primary's 404.
        alt_ready, _ = _get(fallback_url)
        return alt_ready, "" if alt_ready else reason

    _say(f"⏳ Waiting for LLM backend ({backend}) at {health_url} …\n")

    while True:
        ready, permanent = await loop.run_in_executor(None, _check)
        if ready:
            _say("✅ LLM backend is ready.\n")
            return
        if permanent:
            raise RuntimeError(f"LLM backend unreachable: {permanent}")

        now = loop.time()
        if now >= deadline:
            raise RuntimeError(
                f"LLM backend did not become ready within {timeout} s "
                f"(health URL: {health_url})"
            )
        if now - last_progress >= progress_interval:
            elapsed = int(now - (deadline - timeout))
            _say(f"⏳ Still waiting for LLM backend … ({elapsed}s elapsed)\n")
            last_progress = now

        await asyncio.sleep(poll)


async def prewarm_model(model: str) -> None:
    """Ask the backend to load the model now, off the first query's clock."""
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, lambda: get_backend().prewarm(model))
    except Exception:
        # Non-fatal: the first real query loads it.
        logger.debug("prewarm_model: a best-effort step failed and was skipped", exc_info=True)

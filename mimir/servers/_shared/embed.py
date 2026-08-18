"""Backend-aware text embedding with a lexical fallback.

Shared by the MCP servers (which run in separate processes and import ``_shared``
flat via ``sys.path``) and by the client policy layer. It reads the *same*
environment variables as the chat backend — ``LLM_BACKEND``, ``VLLM_BASE_URL``,
``VLLM_API_KEY`` — so it behaves identically on both sides without importing any
client config.

Design contract: **every entry point degrades gracefully.** ``embed_texts``
returns ``None`` on any failure (backend down, no embedding model served, timeout,
missing dependency), and callers fall back to lexical overlap scoring. This keeps
the hermetic pytest suite and embedding-less environments fully working.

Environment:
  LLM_BACKEND           "vllm" | "ollama" (default "vllm", matching the chat side)
  MIMIR_EMBED_MODEL     embedding model name. Required for vLLM (the served model
                        name, e.g. "BAAI/bge-m3"); defaults to "nomic-embed-text"
                        for Ollama.
  MIMIR_EMBED_BASE_URL  optional override so embeddings can be served by a separate
                        endpoint from the chat model; falls back to VLLM_BASE_URL.
  MIMIR_EMBED_TIMEOUT   HTTP timeout in seconds for the vLLM path (default 10).
"""

from __future__ import annotations

import functools
import os
import re
from typing import Any

_DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


# ── configuration ───────────────────────────────────────────────────────────────

def _backend() -> str:
    return os.environ.get("LLM_BACKEND", "vllm").strip().lower()


def _embed_model() -> str:
    return os.environ.get("MIMIR_EMBED_MODEL", "").strip()


def embed_model_id() -> str | None:
    """Resolved embedding model name for the active backend, or ``None`` when the
    vLLM backend has no model configured. Used to key/validate cached vectors so a
    model change invalidates stale embeddings.
    """
    backend = _backend()
    model = _embed_model()
    if backend == "vllm":
        return model or None
    return model or _DEFAULT_OLLAMA_EMBED_MODEL


def _timeout() -> float:
    try:
        return float(os.environ.get("MIMIR_EMBED_TIMEOUT", "10"))
    except ValueError:
        return 10.0


def _resolve_host(base_url: str) -> str:
    """Replace 127.0.0.1/localhost with the node's real hostname so HPC
    reverse-proxies that intercept loopback addresses are bypassed.

    Single source of truth — the vLLM chat backend re-exports this.
    """
    import socket
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(base_url)
    if parsed.hostname in ("127.0.0.1", "localhost"):
        node_host = socket.getfqdn()
        if node_host in ("localhost", "localhost.localdomain"):
            node_host = socket.gethostname()
        netloc = f"{node_host}:{parsed.port}" if parsed.port else node_host
        parsed = parsed._replace(netloc=netloc)
        base_url = urlunparse(parsed)
    return base_url


def verify_ssl() -> bool:
    """Whether to verify TLS certs when talking to the vLLM endpoint.

    Internal corporate routes (e.g. ``https://…​.corp.local/``) are often served
    behind an OpenShift/ingress cert signed by a private CA that isn't in the
    default trust store, so verification fails with ``CERTIFICATE_VERIFY_FAILED``.
    Defaults to **off** — consistent with the ``trust_env=False`` posture used
    everywhere here (the endpoint is an internal cluster service, proxy already
    bypassed), so vLLM works out of the box. Set ``VLLM_VERIFY_SSL=1`` (or
    true/yes/on) to re-enable verification when the endpoint has a publicly
    trusted cert. Single source of truth — the chat backend and embed helper both
    consult this so behaviour is identical on both sides.
    """
    val = os.environ.get("VLLM_VERIFY_SSL", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


# ── embedding backends ──────────────────────────────────────────────────────────

def _embed_vllm(texts: list[str], model: str) -> list[list[float]]:
    """POST to the OpenAI-compatible /v1/embeddings endpoint served by vLLM."""
    import httpx

    base = os.environ.get("MIMIR_EMBED_BASE_URL") or os.environ.get(
        "VLLM_BASE_URL", "http://127.0.0.1:8000"
    )
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    base = _resolve_host(base)
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    url = base.rstrip("/") + "/embeddings"

    headers = {}
    if api_key and api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {api_key}"

    with httpx.Client(trust_env=False, timeout=_timeout(), verify=verify_ssl()) as client:
        resp = client.post(url, json={"model": model, "input": texts}, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    return [list(it["embedding"]) for it in items]


def _embed_ollama(texts: list[str], model: str) -> list[list[float]]:
    """Embed via the local Ollama server (same package the chat backend uses)."""
    import ollama

    resp = ollama.embed(model=model, input=texts)
    embs = resp.get("embeddings") if isinstance(resp, dict) else getattr(resp, "embeddings", None)
    if embs is None:
        raise ValueError("ollama.embed returned no embeddings")
    return [list(e) for e in embs]


# ── public API ──────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Return one embedding vector per input text, or ``None`` on any failure.

    Never raises: a ``None`` return is the signal for callers to fall back to
    lexical scoring.
    """
    if not texts:
        return []
    backend = _backend()
    model = _embed_model()
    try:
        if backend == "vllm":
            if not model:
                # No served embedding model name to target — cannot guess it.
                return None
            return _embed_vllm(texts, model)
        return _embed_ollama(texts, model or _DEFAULT_OLLAMA_EMBED_MODEL)
    except Exception:
        return None


def embed_one(text: str) -> list[float] | None:
    """Convenience wrapper returning a single vector (or ``None``)."""
    vecs = embed_texts([text])
    if not vecs:
        return None
    return vecs[0]


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """Whether the embedding backend answered at least once this process.

    Memoized so a session without embeddings does not re-ping on every call.
    """
    return bool(embed_texts(["ping"]))


def cosine_rank(query_vec: list[float], cand_vecs: list[list[float]]) -> list[tuple[int, float]]:
    """Rank candidate vectors by cosine similarity to *query_vec*, descending.

    Returns ``[(original_index, score), ...]`` with ties broken by original index
    for deterministic output.
    """
    import numpy as np

    q = np.asarray(query_vec, dtype=float)
    qn = float(np.linalg.norm(q))
    if qn == 0.0:
        return [(i, 0.0) for i in range(len(cand_vecs))]
    q = q / qn

    scored: list[tuple[int, float]] = []
    for i, v in enumerate(cand_vecs):
        vec = np.asarray(v, dtype=float)
        vn = float(np.linalg.norm(vec))
        sim = float(q @ vec / vn) if vn else 0.0
        scored.append((i, sim))
    scored.sort(key=lambda s: (-s[1], s[0]))
    return scored


def relevance_tokens(text: str) -> set[str]:
    """Lowercase word/identifier tokens of length ≥3 used for overlap scoring."""
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(t) >= 3}


def lexical_rank(query: str, texts: list[str]) -> list[tuple[int, int]]:
    """Rank texts by token-overlap with the query, descending.

    Returns ``[(original_index, overlap_count), ...]`` with ties broken by original
    index. This is the deterministic fallback shared by both call sites.
    """
    qtok = relevance_tokens(query)
    scored = [(i, len(qtok & relevance_tokens(t))) for i, t in enumerate(texts)]
    scored.sort(key=lambda s: (-s[1], s[0]))
    return scored


def _reset_availability_cache() -> None:
    """Test hook: clear the memoized backend-availability probe."""
    is_available.cache_clear()

"""Backend-aware text embedding, with ``None`` (lexical fallback) on any failure.

Reads the chat backend's variables (``LLM_BACKEND``, ``VLLM_*``, ``RAY_*``) plus
``MIMIR_EMBED_MODEL`` (required for vLLM/Ray), ``MIMIR_EMBED_BASE_URL`` and
``MIMIR_EMBED_TIMEOUT``.
"""

from __future__ import annotations

import functools
import os
import re

_DEFAULT_OLLAMA_EMBED_MODEL = "nomic-embed-text"


# ── configuration ───────────────────────────────────────────────────────────────

def _backend() -> str:
    return os.environ.get("LLM_BACKEND", "vllm").strip().lower()


def _embed_model() -> str:
    return os.environ.get("MIMIR_EMBED_MODEL", "").strip()


# Backends reached over the OpenAI-compatible /v1/embeddings route. Ray Serve is one:
# the router speaks the same API as vLLM, at its own address.
_OPENAI_BACKENDS = ("vllm", "ray")


def embed_model_id() -> str | None:
    """Resolved embedding model name for the active backend, or ``None`` when an
    OpenAI-compatible backend has no model configured. Used to key/validate cached
    vectors so a model change invalidates stale embeddings.
    """
    backend = _backend()
    model = _embed_model()
    if backend in _OPENAI_BACKENDS:
        return model or None
    return model or _DEFAULT_OLLAMA_EMBED_MODEL


def _timeout() -> float:
    try:
        return float(os.environ.get("MIMIR_EMBED_TIMEOUT", "10"))
    except ValueError:
        return 10.0


def verify_ssl() -> bool:
    """Whether to verify TLS for an OpenAI-compatible endpoint (``VLLM_VERIFY_SSL``, default off).

    Internal endpoints often use a private CA; the chat backend and the embedder both read
    this one switch.
    """
    val = os.environ.get("VLLM_VERIFY_SSL", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


# ── embedding backends ──────────────────────────────────────────────────────────

def _embed_vllm(texts: list[str], model: str) -> list[list[float]]:
    """POST to ``/v1/embeddings`` at ``MIMIR_EMBED_BASE_URL``, else the active backend's address."""
    import httpx

    ray = _backend() == "ray"
    default_url = os.environ.get("RAY_BASE_URL" if ray else "VLLM_BASE_URL", "http://127.0.0.1:8000")
    base = os.environ.get("MIMIR_EMBED_BASE_URL") or default_url
    api_key = os.environ.get("RAY_API_KEY" if ray else "VLLM_API_KEY", "EMPTY")
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
        if backend in _OPENAI_BACKENDS:
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

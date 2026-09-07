"""A JSON-backed embedding cache with semantic ranking and a lexical fallback.

Extracted from ``agent_state/server_memory``, which grew the pattern first: a file
of ``{key: {"model": ..., "vec": [...]}}`` beside a corpus, vectors backfilled on the
fly when they are missing or were computed under a different embedding model, and a
caller that degrades to a non-semantic ordering whenever the backend is down.

The one behaviour that is *not* memory's: :func:`semantic_rank` takes an already
narrowed candidate list. Memory embeds its whole corpus eagerly because it holds at
most fifty entries; a corpus of thousands must be prefiltered by the caller, or the
first search pays for embedding the entire thing.

Never raises: every failure path degrades to ``None`` (rank) or a no-op (write), so a
read-only state dir or an unreachable embedding endpoint costs a fallback, not an error.
"""

import json
import os
from typing import Any, Callable, Iterable

import embed as _embed


def load_vectors(path: str) -> dict:
    """Read a vector cache, or {} when absent, unreadable or corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_vectors(path: str, store: dict) -> None:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(store, f)
    except OSError:
        pass


def prune_vectors(path: str, keys: Iterable[str]) -> None:
    """Drop cached vectors for *keys*, rewriting only if something was there."""
    keys = set(keys)
    if not keys:
        return
    store = load_vectors(path)
    if any(k in store for k in keys):
        for k in keys:
            store.pop(k, None)
        save_vectors(path, store)


def semantic_rank(
    query: str,
    items: list,
    *,
    key_of: Callable[[Any], str],
    text_of: Callable[[Any], str],
    path: str,
    limit: int,
) -> list[tuple[Any, float]] | None:
    """Rank *items* by embedding similarity to *query*.

    Returns ``[(item, score), ...]`` best first, or ``None`` to tell the caller to
    fall back — the backend is unavailable, nothing could be embedded, or the query
    itself failed to embed. Vectors absent from the cache, or stored under a
    different embedding model, are recomputed and persisted along the way.
    """
    if not _embed.is_available():
        return None
    store = load_vectors(path)
    model = _embed.embed_model_id()

    vecs: list = []
    ranked_items: list = []
    missing: list = []
    for item in items:
        rec = store.get(key_of(item))
        if rec and rec.get("model") == model and rec.get("vec"):
            vecs.append(rec["vec"])
            ranked_items.append(item)
        else:
            missing.append(item)

    if missing:
        new_vecs = _embed.embed_texts([text_of(item) for item in missing])
        if new_vecs:
            for item, vec in zip(missing, new_vecs):
                store[key_of(item)] = {"model": model, "vec": vec}
                vecs.append(vec)
                ranked_items.append(item)
            save_vectors(path, store)

    if not vecs:
        return None
    qvec = _embed.embed_one(query)
    if qvec is None:
        return None

    return [(ranked_items[pos], round(score, 4))
            for pos, score in _embed.cosine_rank(qvec, vecs)[:limit]]

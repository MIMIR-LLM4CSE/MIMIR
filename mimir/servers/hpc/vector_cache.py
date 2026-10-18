"""A JSON-backed embedding cache with semantic ranking and a lexical fallback.

Vectors are backfilled when missing or from another model; callers prefilter large
corpora. Never raises.
"""

import json
from typing import Any, Callable, Iterable

from .._shared import embed as _embed
from .._shared.atomic import write_json


def load_vectors(path: str) -> dict:
    """Read a vector cache, or {} when absent, unreadable or corrupt."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_vectors(path: str, store: dict) -> None:
    """Replace the cache at *path* atomically: several sessions write it at once."""
    try:
        write_json(path, store)
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
    """``[(item, score)]`` best first by similarity to *query*, or None to fall back."""
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

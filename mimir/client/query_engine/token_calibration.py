"""Token-accounting calibration that survives a restart.

Two figures make the context bar exact rather than merely close, and both are
measured against what the server reported for a real prompt: the fixed per-call
overhead (system prompt + tools schema) and the history's chars-per-token. Measured
in the backend singleton, they used to live and die with the process — so a session
reopened against a fresh server was accounted for by the default heuristic and a
ceiling estimate of the overhead, and the bar moved on its own as soon as the first
answer landed. An endpoint that serves no ``/tokenize`` has no other way back to an
exact number: ``usage.prompt_tokens`` is the only one anyone here gets.

What may be remembered is bounded by what stays true across a restart:

* **chars-per-token** is a property of the model's tokenizer and the kind of text a
  coding session holds, so it is keyed by model alone.
* **prompt overhead** is a property of the *whole fixed part* — model, context mode,
  the system prompt as the mode builds it, and the advertised tools — so it is keyed
  by a fingerprint of all four. Enable a server, switch mode or edit the system
  prompt and the key no longer matches: the entry is simply not found and the caller
  falls back to its estimate. That is the point of keying it this way rather than by
  model: a remembered overhead is either exactly the one that was measured, or absent.
  A stale figure presented as measured would be worse than the honest estimate.

Best-effort throughout: a cache that cannot be read or written costs the session its
head start, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

from ..config import constants

logger = logging.getLogger(__name__)

_FILE = "token_calibration.json"
_VERSION = 1
# Fingerprinted overhead entries to keep. One per (model, mode, prompt, tools)
# combination actually used — a handful in practice, and the oldest go first.
_MAX_OVERHEAD_ENTRIES = 32

_lock = threading.Lock()
_cache: dict | None = None


def _path() -> str:
    """Where the cache lives, resolved at call time.

    Read from the module rather than bound at import: ``STATE_DIR`` is where this
    install keeps its state, and a test that moves it expects the cache to move with
    it rather than to reach into the user's own.
    """
    return os.path.join(constants.STATE_DIR, _FILE)


def _load_locked() -> dict:
    """The cache file as a dict, read once per process. Never raises."""
    global _cache
    if _cache is not None:
        return _cache
    data: dict = {}
    try:
        with open(_path(), "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and raw.get("version") == _VERSION:
            data = raw
    except FileNotFoundError:
        pass
    except Exception:
        logger.debug("token calibration: unreadable cache, starting empty", exc_info=True)
    data.setdefault("version", _VERSION)
    data.setdefault("chars_per_token", {})
    data.setdefault("prompt_overhead", {})
    _cache = data
    return _cache


def _save_locked() -> None:
    """Write the cache out atomically. Never raises."""
    if _cache is None:
        return
    path = _path()
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_cache, fh)
        os.replace(tmp, path)
    except Exception:
        logger.debug("token calibration: could not write cache", exc_info=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass


def overhead_key(model: str, mode: str, system_prompt: str, tools_json: str) -> str:
    """Fingerprint of everything the per-call overhead is made of.

    Hashed rather than stored: the system prompt is tens of thousands of characters,
    and the only question ever asked of it is whether it is the same one.
    """
    digest = hashlib.sha1(
        f"{system_prompt}\x00{tools_json}".encode("utf-8", "replace")
    ).hexdigest()[:16]
    return f"{model}|{mode or 'full'}|{digest}"


def recall_overhead(key: str) -> int | None:
    """The overhead measured for this exact fixed part, or None."""
    with _lock:
        value = _load_locked()["prompt_overhead"].get(key)
    return value if isinstance(value, int) and value > 0 else None


def remember_overhead(key: str, tokens: int) -> None:
    """Record a server-measured overhead for this fixed part."""
    if not key or tokens <= 0:
        return
    with _lock:
        entries = _load_locked()["prompt_overhead"]
        if entries.get(key) == tokens:
            return  # already on disk — the bar asks this once a second
        entries.pop(key, None)  # re-insert so the eviction order below is by recency
        entries[key] = tokens
        while len(entries) > _MAX_OVERHEAD_ENTRIES:
            entries.pop(next(iter(entries)))
        _save_locked()


def recall_chars_per_token(model: str) -> float | None:
    """The chars-per-token ratio measured for *model*, or None."""
    with _lock:
        value = _load_locked()["chars_per_token"].get(model)
    return float(value) if isinstance(value, (int, float)) and value > 0 else None


def remember_chars_per_token(model: str, ratio: float) -> None:
    """Record the chars-per-token ratio measured for *model*."""
    if not model or ratio <= 0:
        return
    with _lock:
        entries = _load_locked()["chars_per_token"]
        # Rounded before the comparison: the ratio drifts in the fourth decimal from
        # one call to the next, which would otherwise rewrite the file every turn.
        value = round(float(ratio), 3)
        if entries.get(model) == value:
            return
        entries[model] = value
        _save_locked()


def reset_for_tests() -> None:
    """Drop the in-process copy so a test can point STATE_DIR somewhere else."""
    global _cache
    with _lock:
        _cache = None

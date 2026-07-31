"""Abstract LLM backend interface."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable

from ...config.constants import chars_per_token_for

# Cap on the per-text token-count cache so a long-running session can't grow it
# unbounded. When exceeded the cache is cleared wholesale (simple + adequate;
# entries are cheap to recompute).
_TOKEN_CACHE_CAP = 8192


def _countable_text(message: dict) -> str:
    """Everything in *message* that is sent to the model, as one string.

    Content alone is not the message: in full-context mode history keeps the
    structured transcript, where an assistant turn carries its ``tool_calls`` and
    usually has empty content. Counting content only scored those turns at ~0 —
    including calls whose arguments hold a whole file — so the context bar
    under-reported and the pre-query trim under-trimmed. Serialising the call
    payload is an approximation of the server's chat template, but a far closer
    one than dropping it.
    """
    parts = [str(message.get("content") or "")]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        try:
            parts.append(json.dumps(tool_calls, default=str))
        except Exception:
            parts.append(str(tool_calls))
    thinking = message.get("thinking")
    if thinking:
        parts.append(str(thinking))
    return "\n".join(p for p in parts if p)


class LLMBackend(ABC):
    def __init__(self) -> None:
        # Maps (model, len, hash(text)) -> token count. Shared across the process
        # via the cached backend singleton, so a count made in the agent worker
        # thread is reused by the front-end's (non-blocking) budget checks.
        self._token_cache: dict[tuple, int] = {}
        # Maps model -> context window (tokens), populated lazily by subclasses.
        self._ctx_window_cache: dict[str, int | None] = {}

    # ── Context window ──────────────────────────────────────────────────────────

    def context_window(self, model: str) -> int | None:
        """Return the model's effective context window (tokens), or None.

        Used so the token-usage bar and history trim/compaction track the real
        window instead of a static assumption. Result is cached per model. The
        default returns None (window unknown); each backend overrides
        :meth:`_fetch_context_window` to report the actual size.
        """
        if model in self._ctx_window_cache:
            return self._ctx_window_cache[model]
        try:
            win = self._fetch_context_window(model)
        except Exception:
            win = None
        self._ctx_window_cache[model] = win
        return win

    def _fetch_context_window(self, model: str) -> int | None:
        """Backend-specific context-window lookup. Default: unknown."""
        return None

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        thinking: bool,
        streaming: bool,
        options: dict,
        cancel_flag: Any = None,
        token_callback: Callable[[str], None] | None = None,
        think_token_callback: Callable[[str], None] | None = None,
        think_start_callback: Callable[[], None] | None = None,
        think_end_callback: Callable[[], None] | None = None,
    ) -> dict:
        """Run a chat completion and return a message dict.

        Return value keys: ``role``, ``content``, and optionally ``thinking``
        and ``tool_calls``.
        """

    # ── Token counting ──────────────────────────────────────────────────────────

    def count_text_tokens(self, model: str, text: str, allow_network: bool = True) -> int:
        """Token count for *text*, cached.

        Exact when the backend has a tokenizer (vLLM /tokenize); otherwise the
        chars-per-token heuristic. ``allow_network=False`` forbids any blocking
        tokenizer round-trip — used from async event loops that must not block:
        a cached value is returned if present, else the heuristic (not cached, so
        it never masks a later exact count).
        """
        if not text:
            return 0
        key = (model, len(text), hash(text))
        cached = self._token_cache.get(key)
        if cached is not None:
            return cached
        if not allow_network:
            return self._heuristic_tokens(model, text)
        try:
            count = self._tokenize_text(model, text)
        except Exception:
            return self._heuristic_tokens(model, text)  # transient failure: don't cache
        if len(self._token_cache) >= _TOKEN_CACHE_CAP:
            self._token_cache.clear()
        self._token_cache[key] = count
        return count

    def message_token_counts(
        self, model: str, messages: list[dict], allow_network: bool = True
    ) -> list[int]:
        """Per-message token counts, aligned with *messages*."""
        return [
            self.count_text_tokens(model, _countable_text(m), allow_network=allow_network)
            for m in messages
        ]

    def count_messages_tokens(
        self, model: str, messages: list[dict], allow_network: bool = True
    ) -> int:
        """Total token count across the content of *messages*."""
        return sum(self.message_token_counts(model, messages, allow_network=allow_network))

    def _tokenize_text(self, model: str, text: str) -> int:
        """Exact token count for *text*. Default: heuristic; vLLM overrides."""
        return self._heuristic_tokens(model, text)

    @staticmethod
    def _heuristic_tokens(model: str, text: str) -> int:
        return max(1, int(len(text) / chars_per_token_for(model)))

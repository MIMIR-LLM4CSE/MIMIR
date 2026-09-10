"""Abstract LLM backend interface."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Callable

from ...config.constants import chars_per_token_for

# Cap on the per-text token-count cache so a long-running session can't grow it
# unbounded. When exceeded the cache is cleared wholesale (simple + adequate;
# entries are cheap to recompute).
class PromptTooLongError(ValueError):
    """The prompt does not fit the model's window, as the provider counted it.

    A ``ValueError`` subclass so nothing that used to catch the untyped error
    stops catching it, and a type of its own so the retry loop can tell the one
    failure that re-sending cannot fix from the flaky connection it looks like.
    Retrying it spent three backoffs and three requests on a prompt that was the
    same size each time, and then ended the query anyway.
    """


_TOKEN_CACHE_CAP = 8192


_FINISH_REASONS = {
    # OpenAI / vLLM / Ray
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
    # Ollama (done_reason)
    "load": "stop",
    "unload": "stop",
    # Anthropic (stop_reason)
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
    "model_context_window_exceeded": "length",
}


def normalize_finish_reason(raw: Any) -> str | None:
    """Map a provider's stop signal onto one vocabulary, or None when absent.

    Returns one of ``stop``, ``length``, ``tool_calls``, ``content_filter`` or
    ``unknown``. ``None`` means the provider said nothing — distinct from
    ``unknown``, which means it said something this table does not recognize.
    """
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if not key:
        return None
    return _FINISH_REASONS.get(key, "unknown")


# The keys a message actually travels with. Serialising the dict wholesale would
# also count local bookkeeping a backend never sees, so the wire form is spelled
# out rather than inferred — an extra key added here later must be one the
# provider receives.
_WIRE_KEYS = ("role", "content", "name", "tool_call_id", "tool_calls", "thinking")


def message_wire_form(message: dict) -> str:
    """*message* as the provider is sent it: the JSON, envelope included.

    Content alone is not the message: in full-context mode history keeps the
    structured transcript, where an assistant turn carries its ``tool_calls`` and
    usually has empty content. Counting content only scored those turns at ~0 —
    including calls whose arguments hold a whole file — so the context bar
    under-reported and the pre-query trim under-trimmed.

    Serialising content and arguments as bare *text* closed that gap and left a
    second one of the same kind. What a window is measured against is
    ``json.dumps(messages)`` — see the guard in ``VLLMBackend.chat`` — so every
    ``\n`` that serialisation escapes, every quote it doubles and every ``role`` /
    ``tool_call_id`` key it adds is prompt that a text-shaped count never saw. On
    an ordinary tool-heavy history that envelope is ~18% of the total; on
    escape-dense content — HTML, LaTeX, JSON inside JSON, the shapes a web fetch
    returns — it reaches 30%. Measured in the wrong units, a budget cannot be
    conservative by accident: it is optimistic by construction, and it was an
    optimism of exactly this size that let a backstop announce a fit on a prompt
    the provider then rejected.

    Falls back to the joined-text approximation on anything unserialisable. That
    only ever under-counts, which is the old behaviour rather than a new failure.
    """
    # Keyed on presence, not on truth: a message that carries ``content: None``
    # is serialised with ``"content": null`` and paid for, so dropping the falsy
    # ones would reintroduce the same under-count one level down — ~16 characters
    # per assistant turn, and an assistant turn is every other message.
    payload = {k: message[k] for k in _WIRE_KEYS if k in message}
    if not payload:
        return ""
    try:
        return json.dumps(payload, default=str)
    except Exception:
        parts = [str(message.get("content") or "")]
        tool_calls = message.get("tool_calls")
        if tool_calls:
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

    def served_models(self) -> list[str]:
        """Model ids the endpoint reports it is serving, weakest-guarantee first.

        Only endpoints that can enumerate themselves return anything; the default
        empty list is what lets a caller resolve a model without first asking which
        backend is active. Never raises — a failure reads as "nothing to offer".
        """
        return []

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

        Return value keys: ``role``, ``content``, and optionally ``thinking``,
        ``tool_calls`` and ``finish_reason``.

        ``finish_reason`` is why the provider stopped generating, normalized by
        :func:`normalize_finish_reason` — every provider spells it differently
        (OpenAI/vLLM ``finish_reason``, Ollama ``done_reason``, Anthropic
        ``stop_reason``) and all three used to drop it, which left a turn cut off
        at ``max_tokens`` indistinguishable from a model that simply had nothing
        to say. It describes the *call*, not the message: strip it before the
        message is appended to history (``streaming._process_response`` does),
        or it goes back to the provider on the next turn.
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
            self.count_text_tokens(model, message_wire_form(m), allow_network=allow_network)
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

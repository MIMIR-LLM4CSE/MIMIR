"""Abstract LLM backend interface."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC
from typing import Any, Callable

from ...config.limits import CHARS_PER_TOKEN
from ...protocol.messages import message_wire_form
from .. import token_calibration
from ...protocol.messages import merge_consecutive_user_messages, reconcile_tool_pairs
from .tag_parser import ThinkTagParser

class PromptTooLongError(ValueError):
    """The prompt does not fit the model's window, as the provider counted it.

    A ``ValueError`` subclass so nothing that used to catch the untyped error
    stops catching it, and a type of its own so the retry loop can tell the one
    failure that re-sending cannot fix from the flaky connection it looks like.
    Retrying it spent three backoffs and three requests on a prompt that was the
    same size each time, and then ended the query anyway.
    """


# Cap on the per-text token-count cache so a long-running session can't grow it
# unbounded. When exceeded the cache is cleared wholesale (simple + adequate;
# entries are cheap to recompute).
_TOKEN_CACHE_CAP = 8192

# Chars-per-token calibration from the server's own prompt_tokens (see
# LLMBackend._calibrate_chars_per_token). Below this many history tokens the fixed
# part's estimation error outweighs the history itself; outside the bounds, the
# figure is a measurement accident, not a tokenizer.
_CPT_MIN_HISTORY_TOKENS = 2000
_CPT_BOUNDS = (2.0, 8.0)


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


def env_positive_int(name: str) -> int | None:
    """The positive integer environment variable *name* holds, or None."""
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


class ChatTurn:
    """What one model call has produced so far, whichever backend is making it.

    A backend reads its provider's response and hands each piece here — prose,
    reasoning, a tool call, the stop reason, the reported prompt size — and the turn
    routes it: prose and reasoning through the shared ``<think>`` parser to the
    front-end's callbacks, the rest into the message :meth:`result` assembles. One
    place, so a call cut off, cancelled or answered without streaming reads the same
    whichever backend made it.

    Without streaming, reasoning never reaches the callbacks: the whole block is
    emitted once by ``streaming._process_response``, and firing the callbacks too
    showed it twice. Prose still does — it is the answer, once.
    """

    def __init__(
        self, *, streaming: bool, cancel_flag: Any = None,
        token_callback: Callable[[str], None] | None = None,
        think_token_callback: Callable[[str], None] | None = None,
        think_start_callback: Callable[[], None] | None = None,
        think_end_callback: Callable[[], None] | None = None,
    ) -> None:
        self.parser = ThinkTagParser(
            token_callback=token_callback,
            think_token_callback=think_token_callback if streaming else None,
            think_start_callback=think_start_callback if streaming else None,
            think_end_callback=think_end_callback if streaming else None,
        )
        self._echo = token_callback is None
        self._cancel_flag = cancel_flag
        self.tool_calls: list[dict] = []
        self.finish_reason: str | None = None
        self.prompt_tokens = 0
        # Message fields a provider reports beyond the text (``role``, ``name``).
        self.fields: dict[str, Any] = {}

    def check_cancelled(self) -> None:
        """Raise when the user pressed Stop. Called per chunk by a streaming read."""
        if self._cancel_flag is not None and self._cancel_flag.is_set():
            raise asyncio.CancelledError("Cancelled by user")

    def content(self, text: str) -> None:
        """Prose that may carry inline ``<think>`` tags."""
        if text:
            self.parser.feed_content(text)

    def text(self, text: str) -> None:
        """Prose a provider already separated from its reasoning."""
        if text:
            self.parser.feed_text(text)

    def thinking(self, text: str) -> None:
        """A provider's reasoning field that may still carry the answer's head
        (Ollama's ``thinking``)."""
        if text:
            self.parser.feed_thinking(text)

    def reasoning(self, text: str) -> None:
        """A provider's reasoning field, holding nothing but the reasoning."""
        if text:
            self.parser.feed_reasoning(text)

    def stopped(self, raw: Any) -> None:
        """Record why the provider stopped, when it says."""
        normalized = normalize_finish_reason(raw)
        if normalized:
            self.finish_reason = normalized

    def result(self) -> dict:
        """The assistant message this call produced.

        Keys: ``role``, ``content``, and optionally ``thinking``, ``tool_calls``,
        ``finish_reason`` and ``prompt_tokens`` (the last two describe the call and are
        taken off before the message enters history).
        """
        self.parser.finish()
        if self.parser.content_parts and self._echo:
            sys.stdout.write("\n")
            sys.stdout.flush()
        msg: dict = {"role": "assistant", **self.fields,
                     "content": "".join(self.parser.content_parts)}
        if self.parser.thinking_parts:
            msg["thinking"] = "".join(self.parser.thinking_parts)
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        if self.finish_reason:
            msg["finish_reason"] = self.finish_reason
        if self.prompt_tokens > 0:
            msg["prompt_tokens"] = self.prompt_tokens
        return msg


class LLMBackend(ABC):
    def __init__(self) -> None:
        # Maps (model, len, hash(text)) -> token count. Shared across the process
        # via the cached backend singleton, so a count made in the agent worker
        # thread is reused by the front-end's (non-blocking) budget checks.
        self._token_cache: dict[tuple, int] = {}
        # Maps model -> context window (tokens), populated lazily by subclasses.
        self._ctx_window_cache: dict[str, int | None] = {}
        # Maps model -> the per-call prompt overhead the server actually charged,
        # derived from a reported prompt_tokens (see :meth:`note_prompt_usage`).
        self._prompt_overhead: dict[str, int] = {}
        # Maps model -> the history's chars-per-token, measured against the server's
        # prompt_tokens (see :meth:`_calibrate_chars_per_token`).
        self._calibrated_cpt: dict[str, float] = {}

    # ── Prompt overhead calibration ─────────────────────────────────────────────

    def _calibrate_chars_per_token(
        self, model: str, system: dict | None, history: list[dict],
        tools: list[dict] | None, prompt_tokens: int,
    ) -> None:
        """Measure the history's chars-per-token against what the server charged.

        Without a tokenizer every count is the chars-per-token heuristic, and history
        is where a fixed 4 goes wrong: on a DeepSeek session it counted 106,902 for
        118,880 charged, and the ratio drifts from 3.3 to 4.2 from one stretch of a
        conversation to the next. The overhead absorbed that error, so the bar showed
        ~48k of system prompt and tools where the server charged 36k. A /tokenize
        round-trip would fix it, but an OpenAI-compatible router need not serve one.

        The fixed part is where the heuristic holds: a tools schema is JSON of steady
        density (35,024 estimated, 34,888 charged), so system prompt and tools are
        estimated at the default ratio and the rest of the reported size is the
        history. Its ratio is then the whole history's average, not one stretch's.
        Pairing two calls to measure the difference between them was tried and
        dropped: it measures only the last stretch, off by a fifth on real sessions.
        """
        default = float(CHARS_PER_TOKEN)
        fixed_chars = len(message_wire_form(system)) if system else 0
        if tools:
            fixed_chars += len(json.dumps(tools))
        history_tokens = prompt_tokens - int(fixed_chars / default)
        if history_tokens < _CPT_MIN_HISTORY_TOKENS:
            return
        ratio = sum(len(message_wire_form(m)) for m in history) / history_tokens
        low, high = _CPT_BOUNDS
        if low <= ratio <= high:
            self._calibrated_cpt[model] = ratio
            token_calibration.remember_chars_per_token(model, ratio)

    def calibrated_chars_per_token(self, model: str) -> float | None:
        """The history's measured chars-per-token for *model*, or None before any.

        This process's own measurement. A ratio measured in an earlier run is kept in
        the calibration cache and used by :meth:`_heuristic_tokens`, but it is not
        reported here: the two answer different questions, and whether *this* backend
        has calibrated itself yet is the one this method is asked.
        """
        return self._calibrated_cpt.get(model)

    def note_prompt_usage(
        self, model: str, messages: list[dict], prompt_tokens: int,
        tools: list[dict] | None = None,
    ) -> None:
        """Record what the server said the prompt cost, as an overhead per model.

        ``prompt_tokens`` from the response's usage block is the only exact figure
        anyone here has: it is what the server counted, chat template and all. Stored
        as *overhead* — the reported total minus this backend's own count of the
        history that went with it — rather than as the total, because the total is a
        fact about one call and goes stale the moment the answer lands, while the
        overhead (system prompt, tools schema, template scaffolding) is the same on
        the next call and is exactly the part estimated worst.

        Counting the history with the same tokenizer that the bar uses is deliberate:
        whatever that count gets wrong is absorbed into the overhead, so bar and
        server agree exactly at the moment of the call instead of agreeing in
        principle and differing in the number shown. The history's chars-per-token is
        re-measured first (:meth:`_calibrate_chars_per_token`), so what is absorbed
        is the fixed part's estimation error, a few percent, not the history's.

        Called by the agent loop alone (``streaming._calibrate_overhead``), never
        from inside :meth:`chat`: a side call with its own short prompt would
        otherwise overwrite the overhead the bar reads. Backends report the figure
        as ``prompt_tokens`` in the returned message instead.

        Best-effort and never raises — usage is telemetry, not part of the turn.
        """
        try:
            if prompt_tokens <= 0:
                return
            has_system = bool(messages) and messages[0].get("role") == "system"
            history = messages[1:] if has_system else messages
            self._calibrate_chars_per_token(
                model, messages[0] if has_system else None, history, tools, prompt_tokens
            )
            counted = self.count_messages_tokens(model, history, allow_network=False)
            overhead = prompt_tokens - counted
            if overhead > 0:
                self._prompt_overhead[model] = overhead
        except Exception:
            pass

    def measured_prompt_overhead(self, model: str) -> int | None:
        """The calibrated overhead for *model*, or None before any call reported one."""
        return self._prompt_overhead.get(model)

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

    def readiness_urls(self) -> list[str]:
        """URLs whose answer says the endpoint is up, best first; [] for none to wait on.

        A hosted API is up as far as anyone here can tell, so the default waits on
        nothing; a self-hosted server that may still be loading names what to poll.
        """
        return []

    def prewarm(self, model: str) -> None:
        """Load *model* ahead of the first query, where the backend can. Default: none."""

    def served_models(self) -> list[str]:
        """Model ids the endpoint reports it is serving, weakest-guarantee first.

        Only endpoints that can enumerate themselves return anything; the default
        empty list is what lets a caller resolve a model without first asking which
        backend is active. Never raises — a failure reads as "nothing to offer".
        """
        return []

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
        """Run a chat completion and return the assistant message (see :class:`ChatTurn`).

        ``finish_reason`` is why the provider stopped generating, normalized by
        :func:`normalize_finish_reason` — every provider spells it differently
        (OpenAI/vLLM ``finish_reason``, Ollama ``done_reason``, Anthropic
        ``stop_reason``) and all three used to drop it, which left a turn cut off
        at ``max_tokens`` indistinguishable from a model that simply had nothing
        to say. It describes the *call*, not the message: strip it before the
        message is appended to history (``streaming._process_response`` does),
        or it goes back to the provider on the next turn.

        The history is repaired here, for every provider: the tool-call pairing
        (a strict API rejects a dangling pair outright) and adjacent user turns
        merged (several templates require strict alternation). The loop does the same
        before every call; this is the last-mile guard for any caller that doesn't.
        """
        turn = ChatTurn(
            streaming=streaming, cancel_flag=cancel_flag,
            token_callback=token_callback, think_token_callback=think_token_callback,
            think_start_callback=think_start_callback, think_end_callback=think_end_callback,
        )
        messages = merge_consecutive_user_messages(reconcile_tool_pairs(list(messages)))
        self._complete(model, messages, tools, thinking, streaming, options, turn)
        return turn.result()

    def _complete(self, model: str, messages: list[dict], tools: list[dict],
                  thinking: bool, streaming: bool, options: dict, turn: ChatTurn) -> None:
        """Send the request, and hand everything the provider answers to *turn*.

        What every real backend implements. Not abstract only so a test double can
        stand in for the whole call by overriding :meth:`chat` instead.
        """
        raise NotImplementedError

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
        """Exact token count for *text*. Default: none, so the heuristic answers.

        Raises rather than returning the heuristic: a returned count is cached, and a
        cached heuristic would outlive the chars-per-token calibration that later
        corrects it.
        """
        raise NotImplementedError("no tokenizer")

    def _heuristic_tokens(self, model: str, text: str) -> int:
        """Chars-per-token estimate, measured against the server once it can be.

        Falls back through the ratio measured in an earlier run before the static
        default: a session reopened against a fresh server then counts its history the
        way the server charged it, instead of at a 4 that was off by a fifth and moved
        the bar on its own as soon as the first answer re-measured it.
        """
        ratio = (self._calibrated_cpt.get(model)
                 or token_calibration.recall_chars_per_token(model)
                 or CHARS_PER_TOKEN)
        return max(1, int(len(text) / ratio))

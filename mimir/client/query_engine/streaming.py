"""Model round-trip: the single backend call with retry, and response shaping.

``_stream_chat`` runs ONE model call with decorrelated-jitter retry/backoff;
``_process_response`` emits the thinking event and appends the assistant message;
``_to_dict`` normalizes an Ollama/Pydantic response to a plain dict. Extracted
from ``agent_loop.py``.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from .backends.base import PromptTooLongError
from .backends.factory import get_backend
from ..event_sink import emit
from ..config.constants import (
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BASE_DELAY_SECS,
    LLM_RETRY_MAX_DELAY_SECS,
)


def _to_dict(obj: Any) -> dict:
    """Normalize an Ollama response object (Pydantic model or dict) to a plain dict."""
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def _process_response(msg: dict, messages: list[dict], thinking: bool, streamed_thinking: bool = False) -> None:
    """Emit thinking block as a structured event and append the assistant message (without thinking) to history.

    When streamed_thinking is True the thinking tokens were already emitted
    token-by-token, so we do NOT emit the full block again.
    For non-streaming mode (streamed_thinking=False) the block is emitted here.
    """
    if thinking and not streamed_thinking and msg.get("thinking"):
        clean_thinking = msg["thinking"].replace("<think>", "").replace("</think>", "").strip()
        if clean_thinking:
            emit({"type": "thinking", "text": clean_thinking})
    # `thinking` and `finish_reason` describe the call, not the message: sending
    # either back to the provider on the next turn is at best noise and at worst a
    # rejected request. This is the single choke point for all four backends.
    msg_for_history = {k: v for k, v in msg.items()
                       if k not in ("thinking", "finish_reason")}
    messages.append(msg_for_history)


def _note_truncated_turn(msg: dict, step: int | None = None) -> None:
    """Announce a turn the provider cut short because it hit its output cap.

    Purely observational — the loop's behaviour is unchanged. Without it a turn
    truncated at ``max_tokens`` is indistinguishable in the transcript from a
    model that had nothing to say, and both surface as the same "empty turn"
    retry; the status lands in ``transcript.jsonl``, so a session can be told
    apart after the fact instead of guessed at.
    """
    if msg.get("finish_reason") != "length":
        return
    where = f" at step {step}" if step is not None else ""
    emit({"type": "status", "text": (
        f"  ⚠ Output cut short by the model's max_tokens "
        f"(finish_reason=length){where} — this turn is incomplete."
    )})


def _partial_tag_at_end(text: str, tag: str) -> int:
    """Return how many chars at the end of *text* form a prefix of *tag*."""
    for length in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:length]):
            return length
    return 0


class _DraftHold:
    """Buffer for streamed answer prose the loop might still refuse.

    A nudge — or the evidence handback — sends the model back to work, and the
    front-end then drops the draft it was rendering: prose appeared as the answer
    and vanished, which is what the user actually watches. Holding is the only fix
    that does not show it in the first place. A held turn streams nothing; the
    buffer is flushed verbatim the moment the turn proves itself (it called a
    tool), and dropped silently when the loop refuses it. On the accepted-answer
    path nothing is flushed: the final ``answer`` carries the text, post-processed
    (verification report, incomplete-run finalization), so flushing the raw draft
    first would only swap one visible text for another.
    """

    def __init__(self, sink: Any) -> None:
        self._sink = sink
        self._buf: list[str] = []

    def capture(self, delta: str) -> None:
        self._buf.append(delta)

    def flush(self) -> None:
        text = "".join(self._buf)
        self._buf.clear()
        if text:
            self._sink(text)

    def discard(self) -> None:
        self._buf.clear()


# Provider errors that say the *request* is wrong, matched on the text because most
# providers give no type to match on. Deliberately narrow: anything unrecognised stays
# retryable, so an unfamiliar transient failure keeps the behaviour it always had.
_DETERMINISTIC_ERROR_MARKERS = (
    "context window",
    "context length",
    "maximum context",
    "too many tokens",
    "reduce the length of the messages",
    "invalid_request_error",
    "invalid api key",
    "authentication",
    "permission denied",
)


def _is_retryable(exc: Exception) -> bool:
    """Whether re-sending the identical request could plausibly succeed.

    A retry loop that cannot tell "the connection dropped" from "this prompt is
    too long" treats both as bad luck. The second is not bad luck: the prompt is
    byte-identical on every attempt, so the backoff only delays an outcome that
    was already decided — three requests and eight seconds, observed, before a
    query ended on the error it would have ended on immediately.

    Errs toward retrying. A transient failure wrongly called permanent loses a
    whole query; a permanent one wrongly called transient loses a few seconds, and
    that is the trade the default has to respect.
    """
    if isinstance(exc, PromptTooLongError):
        return False
    text = f"{exc}".lower()
    return not any(marker in text for marker in _DETERMINISTIC_ERROR_MARKERS)


def _stream_chat(model: str,
                 messages: list[dict],
                 tools: list[dict],
                 thinking: bool,
                 streaming: bool,
                 options: dict,
                 cancel_flag: Any = None,
                 token_callback: Any = None,
                 think_token_callback: Any = None,
                 think_start_callback: Any = None,
                 think_end_callback: Any = None) -> dict:
    """Run one model round-trip, retrying transient backend failures.

    A single step of a long (up to MAX_AGENT_STEPS) loop should not discard the
    whole query because of one flaky connection / 5xx / rate-limit. We retry the
    call with exponential backoff + jitter. User cancellation raises
    asyncio.CancelledError, which subclasses BaseException (not Exception) and so
    is never swallowed here. The cancel_flag is also checked before each retry so
    a mid-backoff cancel bails immediately.

    Caveat: when streaming, a failure can occur after some tokens were already
    emitted via token_callback; the retry re-streams from the start, so the UI
    may briefly show duplicated partial output on the (rare) retry path. We trade
    that cosmetic glitch for not losing the entire query.
    """
    backend = get_backend()
    last_exc: Exception | None = None
    for attempt in range(LLM_RETRY_ATTEMPTS + 1):
        if cancel_flag is not None and cancel_flag.is_set():
            raise asyncio.CancelledError("Cancelled by user")
        try:
            return backend.chat(
                model=model,
                messages=messages,
                tools=tools,
                thinking=thinking,
                streaming=streaming,
                options=options,
                cancel_flag=cancel_flag,
                token_callback=token_callback,
                think_token_callback=think_token_callback,
                think_start_callback=think_start_callback,
                think_end_callback=think_end_callback,
            )
        except Exception as exc:  # noqa: BLE001 — backend exception types vary by provider
            last_exc = exc
            if attempt >= LLM_RETRY_ATTEMPTS or not _is_retryable(exc):
                break
            delay = min(
                LLM_RETRY_BASE_DELAY_SECS * (2 ** attempt),
                LLM_RETRY_MAX_DELAY_SECS,
            )
            delay += random.uniform(0, delay * 0.25)  # decorrelating jitter
            emit({
                "type": "status",
                "text": f"  ⚠ Model call failed ({type(exc).__name__}); retrying in "
                        f"{delay:.1f}s (attempt {attempt + 1}/{LLM_RETRY_ATTEMPTS})",
            })
            time.sleep(delay)

    if last_exc is not None and not _is_retryable(last_exc):
        emit({
            "type": "status",
            "text": f"  ⚠ Model call failed ({type(last_exc).__name__}); not retried — "
                    f"the same request would fail the same way",
        })
    assert last_exc is not None
    raise last_exc

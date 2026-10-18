"""Model round-trip: the single backend call with retry, and response shaping.

``stream_chat`` runs ONE model call with decorrelated-jitter retry/backoff;
``process_response`` emits the thinking event and appends the assistant message.
"""
from __future__ import annotations

import asyncio
import random
import time
import uuid
from typing import Any

from ..llm.base import PromptTooLongError
from ..protocol.tool_call import as_dict
from ..llm.factory import get_backend
from ..event_sink import emit
from ..config.limits import (
    LLM_RETRY_ATTEMPTS,
    LLM_RETRY_BASE_DELAY_SECS,
    LLM_RETRY_MAX_DELAY_SECS,
)


def _ensure_unique_call_ids(msg: dict, messages: list[dict]) -> None:
    """Give every tool call of *msg* an id no earlier call of the conversation has."""
    calls = msg.get("tool_calls")
    if not calls:
        return
    taken = {tc.get("id") for m in messages if m.get("role") == "assistant"
             for tc in (m.get("tool_calls") or []) if isinstance(tc, dict)}
    # Ollama's calls can be SDK objects; a dict is what history and dispatch hold.
    calls = msg["tool_calls"] = [tc if isinstance(tc, dict) else as_dict(tc)
                                 for tc in calls]
    for tc in calls:
        call_id = tc.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in taken:
            call_id = tc["id"] = f"call_{uuid.uuid4().hex[:12]}"
        taken.add(call_id)


def process_response(msg: dict, messages: list[dict], thinking: bool, streamed_thinking: bool = False) -> None:
    """Emit thinking block as a structured event and append the assistant message (without thinking) to history.

    When streamed_thinking is True the thinking tokens were already emitted
    token-by-token, so we do NOT emit the full block again.
    For non-streaming mode (streamed_thinking=False) the block is emitted here.
    """
    _ensure_unique_call_ids(msg, messages)
    if thinking and not streamed_thinking and msg.get("thinking"):
        clean_thinking = msg["thinking"].replace("<think>", "").replace("</think>", "").strip()
        if clean_thinking:
            emit({"type": "thinking", "text": clean_thinking})
    # `thinking` and `finish_reason` describe the call, not the message: sending
    # either back to the provider on the next turn is at best noise and at worst a
    # rejected request. This is the single choke point for all four backends.
    msg_for_history = {k: v for k, v in msg.items()
                       if k not in ("thinking", "finish_reason")}
    # Trailing blank lines carry nothing, and replayed they grow: in one poll loop each
    # step copied the previous one and added two more newlines, 22 steps in a row. The
    # history keeps the text, not the ramp.
    if isinstance(msg_for_history.get("content"), str):
        msg_for_history["content"] = msg_for_history["content"].rstrip()
    # The reasoning itself is part of the message, though. Interleaved-thinking models
    # (DeepSeek-V4 with tools, GLM, Qwen3 within a turn) render each earlier step as
    # `<think>{reasoning}</think>{content}{calls}`; stripped, every step showed an empty
    # block, so the model re-derived the same analysis from the same tool results and
    # restated it before each call. Kept under `reasoning`, the field vLLM reads back;
    # the chat template decides what of it the prompt shows.
    reasoning = (msg.get("thinking") or "").replace("<think>", "").replace("</think>", "").strip()
    if reasoning:
        msg_for_history["reasoning"] = reasoning
    messages.append(msg_for_history)


def note_truncated_turn(msg: dict, step: int | None = None) -> None:
    """Announce a turn the provider cut at its output cap (a status in the transcript)."""
    if msg.get("finish_reason") != "length":
        return
    where = f" at step {step}" if step is not None else ""
    emit({"type": "status", "text": (
        f"  ⚠ Output cut short by the model's max_tokens "
        f"(finish_reason=length){where} — this turn is incomplete."
    )})


class DraftHold:
    """Buffer for streamed prose the loop might still refuse.

    Flushed when the turn calls a tool, dropped when refused; an accepted answer arrives
    in the final ``answer`` instead.
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
    """Whether re-sending the identical request could succeed; errs toward yes."""
    if isinstance(exc, PromptTooLongError):
        return False
    text = f"{exc}".lower()
    return not any(marker in text for marker in _DETERMINISTIC_ERROR_MARKERS)


def _calibrate_overhead(backend: Any, model: str, messages: list[dict], msg: Any,
                        tools: list[dict] | None = None) -> None:
    """Hand the server's reported prompt size to the context bar's calibration (loop calls only).

    Side calls (summaries, compaction) send prompts of their own and would skew it.
    """
    if not isinstance(msg, dict):
        return
    reported = msg.pop("prompt_tokens", None)
    note = getattr(backend, "note_prompt_usage", None)
    if reported and callable(note):
        note(model, messages, reported, tools=tools)


def stream_chat(model: str,
                 messages: list[dict],
                 tools: list[dict],
                 thinking: bool,
                 streaming: bool,
                 options: dict,
                 cancel_flag: Any = None,
                 token_callback: Any = None,
                 think_token_callback: Any = None,
                 think_start_callback: Any = None,
                 think_end_callback: Any = None,
                 on_retry: Any = None) -> dict:
    """One model round-trip, retrying transient failures with backoff and jitter.

    Cancellation propagates, also mid-backoff. ``on_retry`` runs before each new attempt so
    a held draft can drop the failed attempt's text.
    """
    backend = get_backend()
    last_exc: Exception | None = None
    for attempt in range(LLM_RETRY_ATTEMPTS + 1):
        if cancel_flag is not None and cancel_flag.is_set():
            raise asyncio.CancelledError("Cancelled by user")
        try:
            msg = backend.chat(
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
            _calibrate_overhead(backend, model, messages, msg, tools)
            return msg
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
            if on_retry is not None:
                on_retry()
            # A Stop during a 30 s backoff must not wait it out.
            if cancel_flag is not None and hasattr(cancel_flag, "wait"):
                cancel_flag.wait(delay)
            else:
                time.sleep(delay)

    if last_exc is not None and not _is_retryable(last_exc):
        emit({
            "type": "status",
            "text": f"  ⚠ Model call failed ({type(last_exc).__name__}); not retried — "
                    f"the same request would fail the same way",
        })
    assert last_exc is not None
    raise last_exc

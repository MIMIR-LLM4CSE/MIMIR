"""Reusable scripted LLM backend for agent-loop tests.

``ScriptedBackend`` is a deterministic ``LLMBackend`` whose ``chat()`` replays a
pre-built list of response dicts (one popped per call). It lets a test drive the
real agent loop end-to-end without a live model: feed it a script of
``{"content": ...}`` / ``{"tool_calls": [...]}`` turns and assert on the loop's
behaviour. The same backend powers a zero-cost CI mode for the eval harness.

The response dicts use the canonical backend return shape (see
``LLMBackend.chat``): keys ``role``, ``content``, optional ``thinking`` and
``tool_calls`` (each ``{"id", "function": {"name", "arguments": <json-or-dict>}}``).
"""
from __future__ import annotations

from typing import Any, Callable

from mimir.client.query_engine.backends.base import LLMBackend


class ScriptedBackend(LLMBackend):
    """An ``LLMBackend`` that returns canned ``chat()`` responses in order.

    Parameters
    ----------
    responses:
        Response dicts returned by successive ``chat()`` calls. When exhausted,
        ``chat()`` returns an empty-content assistant message (a natural turn
        end), so a script that runs past its end terminates the loop rather
        than hanging.
    tokenize:
        Optional ``(model, text) -> int`` used by ``_tokenize_text`` (and thus
        ``count_text_tokens``). Defaults to the base chars-per-token heuristic.
    """

    def __init__(
        self,
        responses: list[dict] | None = None,
        tokenize: Callable[[str, str], int] | None = None,
    ) -> None:
        super().__init__()
        self._responses = list(responses or [])
        self._tokenize = tokenize
        # Per-call record of (model, messages, tools, options) for assertions.
        self.calls: list[dict] = []

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
        self.calls.append(
            {
                "model": model,
                "messages": [dict(m) for m in messages],
                "tools": tools,
                "options": dict(options or {}),
                "thinking": thinking,
                "streaming": streaming,
            }
        )

        if self._responses:
            resp = dict(self._responses.pop(0))
        else:
            resp = {"role": "assistant", "content": ""}
        resp.setdefault("role", "assistant")

        # Replay the streaming callbacks so tests exercise the same code path the
        # real backends drive (thinking block, then content tokens).
        if streaming:
            if resp.get("thinking") and think_token_callback is not None:
                if think_start_callback is not None:
                    think_start_callback()
                think_token_callback(resp["thinking"])
                if think_end_callback is not None:
                    think_end_callback()
            if resp.get("content") and token_callback is not None:
                token_callback(resp["content"])

        return resp

    def _tokenize_text(self, model: str, text: str) -> int:
        if self._tokenize is not None:
            return self._tokenize(model, text)
        return super()._tokenize_text(model, text)

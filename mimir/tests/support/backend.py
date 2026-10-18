"""A scripted ``LLMBackend`` for agent-loop tests: ``chat()`` replays canned responses.

Each response uses the backend return shape (``role``, ``content``, ``thinking``,
``tool_calls``) and is returned verbatim, so ``finish_reason`` can be scripted too.
"""
from __future__ import annotations

from typing import Any, Callable

from mimir.client.query_engine.backends.base import LLMBackend


class ScriptedBackend(LLMBackend):
    """An ``LLMBackend`` that returns canned ``chat()`` responses in order.

    Past the end of *responses* it returns an empty answer, which ends the loop. *tokenize*
    is an optional ``(model, text) -> int``; the default is the chars-per-token heuristic.
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
                # Recorded so a test can assert an internal call supplied a token
                # sink — without one the real backends echo to stdout.
                "token_callback": token_callback,
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

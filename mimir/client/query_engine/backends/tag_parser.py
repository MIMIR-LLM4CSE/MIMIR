"""Shared <think>...</think> tag state machine.

Both the Ollama and vLLM backends use this parser to route streaming tokens
to either the content or the thinking output stream.

Usage:
    parser = ThinkTagParser()
    # For each streaming chunk:
    content_out, think_out = parser.feed(chunk, is_thinking_field=False)
    # content_out and think_out are lists of str tokens to emit.
    # Call parser.started and parser.ended to check thinking block state.
"""

from __future__ import annotations

import sys
from typing import Callable


class ThinkTagParser:
    """Stateful parser that routes tokens between content and thinking streams."""

    def __init__(
        self,
        token_callback: Callable[[str], None] | None = None,
        think_token_callback: Callable[[str], None] | None = None,
        think_start_callback: Callable[[], None] | None = None,
        think_end_callback: Callable[[], None] | None = None,
    ) -> None:
        self._token_cb = token_callback
        self._think_token_cb = think_token_callback
        self._think_start_cb = think_start_callback
        self._think_end_cb = think_end_callback

        self._in_think = False
        self._buf = ""
        self._thinking_started = False

        self.content_parts: list[str] = []
        self.thinking_parts: list[str] = []

    # ── public ───────────────────────────────────────────────────────────────

    def feed_content(self, raw: str) -> None:
        """Feed a chunk from the model's *content* field."""
        self._buf += raw
        self._process(default_to_thinking=False)

    def feed_thinking(self, raw: str) -> None:
        """Feed a chunk from the model's *thinking* field (Ollama native).

        When no <think> tag is found the whole chunk is treated as thinking.
        When a <think> tag is found the text before it is treated as content
        (the answer that came after </think> earlier in the stream).
        """
        self._buf += raw
        self._process(default_to_thinking=True)

    # ── internals ────────────────────────────────────────────────────────────

    @staticmethod
    def _partial_tag_at_end(text: str, tag: str) -> int:
        """Return the length of any *partial* prefix of *tag* at the end of *text*."""
        for n in range(1, len(tag)):
            if text.endswith(tag[:n]):
                return n
        return 0

    def _emit_content(self, text: str) -> None:
        if not text:
            return
        if self._token_cb is not None:
            self._token_cb(text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()
        self.content_parts.append(text)

    def _emit_thinking(self, text: str) -> None:
        if not text:
            return
        if not self._thinking_started:
            self._thinking_started = True
            if self._think_start_cb is not None:
                self._think_start_cb()
        if self._think_token_cb is not None:
            self._think_token_cb(text)
        self.thinking_parts.append(text)

    def _process(self, *, default_to_thinking: bool) -> None:
        while self._buf:
            if self._in_think:
                end = self._buf.find("</think>")
                if end == -1:
                    partial = self._partial_tag_at_end(self._buf, "</think>")
                    flush = self._buf[: len(self._buf) - partial]
                    self._buf = self._buf[len(self._buf) - partial :]
                    self._emit_thinking(flush)
                    break
                else:
                    flush = self._buf[:end]
                    self._buf = self._buf[end + len("</think>") :]
                    self._in_think = False
                    self._emit_thinking(flush)
                    if self._think_end_cb is not None:
                        self._think_end_cb()
            else:
                # Strip stray </think> that some models emit as content
                # when thinking is disabled (no matching <think> was seen).
                self._buf = self._buf.replace("</think>", "")
                start = self._buf.find("<think>")
                if start == -1:
                    partial = self._partial_tag_at_end(self._buf, "<think>")
                    flush = self._buf[: len(self._buf) - partial]
                    self._buf = self._buf[len(self._buf) - partial :]
                    if flush:
                        if default_to_thinking:
                            self._emit_thinking(flush)
                        else:
                            self._emit_content(flush)
                    break
                else:
                    flush = self._buf[:start]
                    self._buf = self._buf[start + len("<think>") :]
                    self._in_think = True
                    if flush:
                        # Text before <think> goes to content, even inside
                        # the thinking field (it's the answer prefix).
                        self._emit_content(flush)

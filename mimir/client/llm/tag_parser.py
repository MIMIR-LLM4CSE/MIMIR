"""The ``<think>…</think>`` state machine that routes streamed tokens to content or thinking.

``feed(chunk, is_thinking_field)`` returns ``(content_out, think_out)``; call ``finish()``
at the end of the stream.
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
        self._thinking_ended = False

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

    def feed_text(self, text: str) -> None:
        """Feed prose a provider already separated from its reasoning: never parsed."""
        self._emit_content(text)

    def feed_reasoning(self, text: str) -> None:
        """Feed reasoning a provider sends in its own field (vLLM ``reasoning``).

        Never parsed for tags: the field holds nothing but the reasoning.
        """
        self._emit_thinking(text)

    def finish(self) -> None:
        """End of the stream: release a held partial tag as text, and close an open thinking block once."""
        if self._buf:
            held, self._buf = self._buf, ""
            if self._in_think:
                self._emit_thinking(held)
            else:
                self._emit_content(held)
        self._end_thinking()

    # ── internals ────────────────────────────────────────────────────────────

    def _end_thinking(self) -> None:
        if self._thinking_started and not self._thinking_ended:
            self._thinking_ended = True
            if self._think_end_cb is not None:
                self._think_end_cb()

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
                    self._end_thinking()
            else:
                # Strip stray </think> that some models emit as content when thinking
                # is disabled (no matching <think> was seen) — only ahead of the next
                # <think>: past it, a </think> is that block's own end, and stripping
                # it filed a whole one-chunk answer under thinking.
                start = self._buf.find("<think>")
                head = self._buf if start == -1 else self._buf[:start]
                if "</think>" in head:
                    self._buf = head.replace("</think>", "") + self._buf[len(head):]
                    start = self._buf.find("<think>")
                if start == -1:
                    # Either tag may be split across chunks: a stray "</th" held back
                    # is stripped once its end arrives, instead of leaking as content.
                    partial = max(self._partial_tag_at_end(self._buf, "<think>"),
                                  self._partial_tag_at_end(self._buf, "</think>"))
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

"""Ollama LLM backend — wraps the ollama Python package."""
from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Callable

import ollama

from .base import LLMBackend
from .tag_parser import ThinkTagParser


def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return {}


class OllamaBackend(LLMBackend):
    def _fetch_context_window(self, model: str) -> int | None:
        """Ollama context window = the model's trained context length.

        ``MIMIR_OLLAMA_NUM_CTX`` overrides it (cap KV-cache VRAM, or pin a
        value). Otherwise we read ``<arch>.context_length`` from ``ollama.show``.
        The returned value is also what :meth:`chat` passes as ``num_ctx`` so the
        budget and what Ollama actually allocates stay in agreement.
        """
        env = os.environ.get("MIMIR_OLLAMA_NUM_CTX", "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
        info = _to_dict(ollama.show(model)).get("modelinfo") or {}
        arch = info.get("general.architecture")
        if arch and isinstance(info.get(f"{arch}.context_length"), int):
            return info[f"{arch}.context_length"]
        # Fall back to any key ending in .context_length.
        for key, val in info.items():
            if key.endswith(".context_length") and isinstance(val, int):
                return val
        return None

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
        parser = ThinkTagParser(
            token_callback=token_callback,
            think_token_callback=think_token_callback,
            think_start_callback=think_start_callback,
            think_end_callback=think_end_callback,
        )

        tool_calls_parts: list[dict] = []
        final_msg: dict = {}

        # Ollama defaults num_ctx to a small window (~4K) and silently truncates
        # the prompt. Set it to the model's real context window so it matches the
        # agent's budget — unless the caller already pinned num_ctx.
        if "num_ctx" not in options:
            win = self.context_window(model)
            if win:
                options = {**options, "num_ctx": win}

        answer = ollama.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=thinking,
            stream=streaming,
            options=options,
        )

        if streaming:
            for chunk in answer:
                if cancel_flag is not None and cancel_flag.is_set():
                    raise asyncio.CancelledError("Cancelled by user")

                chunk_msg = _to_dict(chunk.get("message", {}))

                raw_content = chunk_msg.get("content") or ""
                if raw_content:
                    parser.feed_content(raw_content)

                think_delta = chunk_msg.get("thinking") or ""
                if think_delta:
                    parser.feed_thinking(think_delta)

                chunk_tool_calls = chunk_msg.get("tool_calls") or []
                if chunk_tool_calls:
                    tool_calls_parts.extend(chunk_tool_calls)

                if "role" in chunk_msg:
                    final_msg["role"] = chunk_msg["role"]
                if "name" in chunk_msg:
                    final_msg["name"] = chunk_msg["name"]
        else:
            chunk_msg = _to_dict(answer.get("message", {}))
            raw_content = chunk_msg.get("content") or ""
            if raw_content:
                # Use a callback-free parser to separate <think>...</think> from
                # content without firing streaming events — _process_response emits
                # the thinking block after this call returns.
                _p = ThinkTagParser()
                _p.feed_content(raw_content)
                parser.content_parts.extend(_p.content_parts)
                parser.thinking_parts.extend(_p.thinking_parts)
                if _p.content_parts and token_callback is None:
                    sys.stdout.write("".join(_p.content_parts))
                    sys.stdout.flush()

            think_delta = chunk_msg.get("thinking") or ""
            if think_delta:
                parser.thinking_parts.append(think_delta)

            chunk_tool_calls = chunk_msg.get("tool_calls") or []
            if chunk_tool_calls:
                tool_calls_parts.extend(chunk_tool_calls)

            if "role" in chunk_msg:
                final_msg["role"] = chunk_msg["role"]
            if "name" in chunk_msg:
                final_msg["name"] = chunk_msg["name"]

        if parser.content_parts and token_callback is None:
            sys.stdout.write("\n")
            sys.stdout.flush()

        final_msg["content"] = "".join(parser.content_parts)

        if parser.thinking_parts:
            final_msg["thinking"] = "".join(parser.thinking_parts)

        if tool_calls_parts:
            final_msg["tool_calls"] = tool_calls_parts

        return final_msg

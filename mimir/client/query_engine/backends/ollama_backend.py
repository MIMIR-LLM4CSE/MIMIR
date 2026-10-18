"""Ollama LLM backend — wraps the ollama Python package."""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any

try:
    import ollama
except ImportError:  # the backend is optional: its SDK is needed only to call it
    ollama = None

from .base import ChatTurn, LLMBackend, env_positive_int
from ...protocol.tool_call import as_dict


def _done_reason(obj: Any) -> Any:
    """Ollama's stop signal, read off the response envelope (never off ``message``)."""
    if isinstance(obj, dict):
        return obj.get("done_reason")
    return getattr(obj, "done_reason", None)


class OllamaBackend(LLMBackend):
    def readiness_urls(self) -> list[str]:
        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        return [f"{base}/api/tags"]

    def prewarm(self, model: str) -> None:
        """Load *model* into VRAM: ``/api/generate`` with an empty prompt.

        Through an opener that ignores HTTP(S)_PROXY — the server is an internal
        address a corporate proxy has no route to, and ``urlopen`` would otherwise
        consult the environment, unlike the ``trust_env=False`` clients elsewhere.
        """
        base = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        body = json.dumps({"model": model, "prompt": "", "stream": False,
                           "keep_alive": "60m"}).encode()
        request = urllib.request.Request(f"{base}/api/generate", data=body,
                                         headers={"Content-Type": "application/json"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=120):
            pass

    def _fetch_context_window(self, model: str) -> int | None:
        """The model's trained context length (``MIMIR_OLLAMA_NUM_CTX`` overrides), also sent as ``num_ctx``."""
        pinned = env_positive_int("MIMIR_OLLAMA_NUM_CTX")
        if pinned:
            return pinned
        if ollama is None:
            return None
        info = as_dict(ollama.show(model)).get("modelinfo") or {}
        arch = info.get("general.architecture")
        if arch and isinstance(info.get(f"{arch}.context_length"), int):
            return info[f"{arch}.context_length"]
        # Fall back to any key ending in .context_length.
        for key, val in info.items():
            if key.endswith(".context_length") and isinstance(val, int):
                return val
        return None

    def _complete(self, model: str, messages: list[dict], tools: list[dict],
                  thinking: bool, streaming: bool, options: dict, turn: ChatTurn) -> None:
        # Ollama defaults num_ctx to a small window (~4K) and silently truncates
        # the prompt. Set it to the model's real context window so it matches the
        # agent's budget — unless the caller already pinned num_ctx.
        if "num_ctx" not in options:
            win = self.context_window(model)
            if win:
                options = {**options, "num_ctx": win}

        # History keeps an assistant step's reasoning under `reasoning`; Ollama's
        # message schema names the same thing `thinking`.
        messages = [
            {**{k: v for k, v in m.items() if k != "reasoning"}, "thinking": m["reasoning"]}
            if m.get("reasoning") else m
            for m in messages
        ]

        if ollama is None:
            raise RuntimeError("LLM_BACKEND=ollama needs the 'ollama' package installed")
        answer = ollama.chat(
            model=model,
            messages=messages,
            tools=tools,
            think=thinking,
            stream=streaming,
            options=options,
        )
        for chunk in (answer if streaming else [answer]):
            turn.check_cancelled()
            # done_reason rides on the response envelope, not on `message`, and is
            # only set on the terminal chunk.
            turn.stopped(_done_reason(chunk))
            msg = as_dict(chunk.get("message", {}))
            turn.content(msg.get("content") or "")
            turn.thinking(msg.get("thinking") or "")
            turn.tool_calls.extend(msg.get("tool_calls") or [])
            for key in ("role", "name"):
                if key in msg:
                    turn.fields[key] = msg[key]

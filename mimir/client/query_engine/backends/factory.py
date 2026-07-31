"""Backend factory — returns a cached LLMBackend singleton based on LLM_BACKEND env var."""
from __future__ import annotations

import os

from .base import LLMBackend

_singleton: dict[str, LLMBackend] = {}


def get_backend() -> LLMBackend:
    backend_name = os.environ.get("LLM_BACKEND", "ollama").lower()
    if backend_name not in _singleton:
        if backend_name == "vllm":
            from .vllm_backend import VllmBackend
            _singleton[backend_name] = VllmBackend()
        elif backend_name in ("anthropic", "claude"):
            from .anthropic_backend import AnthropicBackend
            _singleton[backend_name] = AnthropicBackend()
        else:
            from .ollama_backend import OllamaBackend
            _singleton[backend_name] = OllamaBackend()
    return _singleton[backend_name]


def clear_backend_cache() -> None:
    """Evict the cached singleton so the next call to get_backend() recreates it."""
    _singleton.clear()

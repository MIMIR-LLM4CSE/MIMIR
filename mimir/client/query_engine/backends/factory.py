"""Backend factory — returns a cached LLMBackend singleton based on LLM_BACKEND env var."""
from __future__ import annotations

import os

from .base import LLMBackend

_singleton: dict[str, LLMBackend] = {}


def get_backend() -> LLMBackend:
    backend_name = os.environ.get("LLM_BACKEND", "vllm").lower()
    if backend_name not in _singleton:
        if backend_name == "vllm":
            from .vllm_backend import VllmBackend
            _singleton[backend_name] = VllmBackend()
        elif backend_name == "ray":
            from .ray_backend import RayBackend
            _singleton[backend_name] = RayBackend()
        elif backend_name in ("anthropic", "claude"):
            from .anthropic_backend import AnthropicBackend
            _singleton[backend_name] = AnthropicBackend()
        else:
            from .ollama_backend import OllamaBackend
            _singleton[backend_name] = OllamaBackend()
    return _singleton[backend_name]


def switch_backend(name: str) -> None:
    """Make *name* the backend every model call uses from now on, in this process.

    The choice lives in ``LLM_BACKEND`` because a backend is process-wide: it is what
    ``get_backend`` reads, and what the MCP servers are spawned with.
    """
    os.environ["LLM_BACKEND"] = name
    clear_backend_cache()


def clear_backend_cache() -> None:
    """Evict the cached singleton so the next call to get_backend() recreates it."""
    _singleton.clear()

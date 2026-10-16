"""Backend factory — one cached LLMBackend per process, chosen by ``LLM_BACKEND``."""
from __future__ import annotations

import importlib
import os

from .base import LLMBackend

# Every backend, by the name LLM_BACKEND gives it: (module, class). Imported on first
# use, so a provider's SDK is needed only when that provider is the one chosen.
BACKENDS: dict[str, tuple[str, str]] = {
    "vllm": (".vllm_backend", "VllmBackend"),
    "ray": (".ray_backend", "RayBackend"),
    "ollama": (".ollama_backend", "OllamaBackend"),
    "anthropic": (".anthropic_backend", "AnthropicBackend"),
    "claude": (".anthropic_backend", "AnthropicBackend"),
}

_cache: dict[str, LLMBackend] = {}


def get_backend() -> LLMBackend:
    """The backend ``LLM_BACKEND`` names (default ``vllm``).

    An unknown name is an error: it used to fall through to Ollama, so a typo talked to
    a server nobody had asked for and failed somewhere far from the cause.
    """
    name = os.environ.get("LLM_BACKEND", "vllm").strip().lower()
    if name not in _cache:
        if name not in BACKENDS:
            raise ValueError(f"unknown LLM_BACKEND {name!r}; expected one of "
                             f"{', '.join(sorted(BACKENDS))}")
        module, cls = BACKENDS[name]
        _cache[name] = getattr(importlib.import_module(module, __package__), cls)()
    return _cache[name]


def switch_backend(name: str) -> None:
    """Make *name* the backend every model call uses from now on, in this process.

    The choice lives in ``LLM_BACKEND`` because a backend is process-wide: it is what
    ``get_backend`` reads, and what the MCP servers are spawned with.
    """
    os.environ["LLM_BACKEND"] = name
    clear_backend_cache()


def clear_backend_cache() -> None:
    """Evict the cached singleton so the next call to get_backend() recreates it."""
    _cache.clear()

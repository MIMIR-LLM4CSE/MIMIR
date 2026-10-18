"""The Ray Serve LLM backend: the OpenAI-compatible router in front of vLLM engines.

The OpenAI-compatible backend at another address (``RAY_BASE_URL``/``RAY_API_KEY``); the
router may serve no ``/tokenize`` and no ``max_model_len``. Speaks HTTP only.
"""
from __future__ import annotations

from ..config.models import RAY_API_KEY, RAY_BASE_URL
from .openai_compat import Endpoint, OpenAICompatBackend

RAY_ENDPOINT = Endpoint(
    url_env="RAY_BASE_URL",
    key_env="RAY_API_KEY",
    default_url=RAY_BASE_URL,
    default_key=RAY_API_KEY,
    # ``max_model_len`` is a vLLM-server field, and a router answering /v1/models with
    # the plain OpenAI shape omits it — this is the fallback that matters most here.
    window_env="MIMIR_RAY_MAX_MODEL_LEN",
)


class RayBackend(OpenAICompatBackend):
    endpoint = RAY_ENDPOINT

    def __init__(self) -> None:
        super().__init__()
        # Set once the router has told us it has no /tokenize (see _tokenize_text).
        self._no_tokenize = False

    def _tokenize_text(self, model: str, text: str) -> int:
        """Exact token count through ``/tokenize``; the first failure latches and later calls raise at once."""
        if self._no_tokenize:
            raise RuntimeError("endpoint has no /tokenize")
        try:
            return super()._tokenize_text(model, text)
        except Exception:
            self._no_tokenize = True
            raise

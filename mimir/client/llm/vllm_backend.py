"""vLLM backend — the OpenAI-compatible API a vLLM server serves."""
from __future__ import annotations

from ..config.models import VLLM_API_KEY, VLLM_BASE_URL
from .openai_compat import Endpoint, OpenAICompatBackend

VLLM_ENDPOINT = Endpoint(
    url_env="VLLM_BASE_URL",
    key_env="VLLM_API_KEY",
    default_url=VLLM_BASE_URL,
    default_key=VLLM_API_KEY,
    window_env="MIMIR_VLLM_MAX_MODEL_LEN",
    serves_health=True,
)


class VllmBackend(OpenAICompatBackend):
    endpoint = VLLM_ENDPOINT

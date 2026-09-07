"""Ray Serve LLM backend — the OpenAI-compatible router in front of vLLM engines.

Ray Serve LLM (``ray.serve.llm`` / ``build_openai_app``) exists to place, scale and
route model replicas across a GPU cluster; the engines it drives are vLLM, and what
it exposes is the same OpenAI API. So this backend is :class:`VllmBackend` pointed at
another address: the request builder, the reasoning profiles, the tool-call
normalisation and the streaming parser are the ones that already match those engines.

What genuinely differs is what the *router* serves, not what the engine does:

- its own address and key (``RAY_BASE_URL`` / ``RAY_API_KEY``);
- ``/tokenize`` is a vLLM-server extension the router does not have to proxy;
- ``max_model_len`` in ``/v1/models`` is likewise the vLLM server's field, so the
  window may have to be pinned by hand.

The ``ray`` package itself is not a dependency here — it runs on the cluster, and
this side only ever speaks HTTP.
"""
from __future__ import annotations

import os

from .vllm_backend import VllmBackend, served_model_len


def _get_ray_config() -> tuple[str, str]:
    """Return (base_url, api_key) for the Ray Serve endpoint."""
    try:
        from ...config.models import RAY_BASE_URL, RAY_API_KEY
        base_url = os.environ.get("RAY_BASE_URL", RAY_BASE_URL)
        api_key = os.environ.get("RAY_API_KEY", RAY_API_KEY)
    except ImportError:
        base_url = os.environ.get("RAY_BASE_URL", "http://127.0.0.1:8000")
        api_key = os.environ.get("RAY_API_KEY", "EMPTY")
    # The openai client appends /chat/completions to base_url, so it must end with
    # /v1 — a Serve app is often mounted under a route prefix, so the user's address
    # may be http://<head>:8000/<route> and the suffix is ours to add.
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"
    return base_url, api_key


class RayBackend(VllmBackend):
    def __init__(self) -> None:
        super().__init__()
        # Set once the router has told us it has no /tokenize (see _tokenize_text).
        self._no_tokenize = False

    def _config(self) -> tuple[str, str]:
        return _get_ray_config()

    def _fetch_context_window(self, model: str) -> int | None:
        """Ray Serve context window — the router's max_model_len, if it reports one.

        ``MIMIR_RAY_MAX_MODEL_LEN`` overrides it, and is the escape hatch that
        matters here: ``max_model_len`` is a vLLM-server field, and a router that
        answers ``/v1/models`` with the plain OpenAI shape omits it. Without a
        window the agent falls back to its static budget, which under-uses a large
        model — so pin it when the endpoint stays silent.
        """
        env = os.environ.get("MIMIR_RAY_MAX_MODEL_LEN", "").strip()
        if env.isdigit() and int(env) > 0:
            return int(env)
        return served_model_len(model, self._config())

    def _tokenize_text(self, model: str, text: str) -> int:
        """Exact token count via /tokenize, while the router still has one.

        ``count_text_tokens`` already treats a raised exception as "use the
        heuristic", so a router without the endpoint is not an error. But it counts
        every message of every turn, and re-attempting a request we know 404s would
        pay a round-trip per count — so the first failure latches the answer and
        every later call raises immediately, off the network.
        """
        if self._no_tokenize:
            raise RuntimeError("endpoint has no /tokenize")
        try:
            return super()._tokenize_text(model, text)
        except Exception:
            self._no_tokenize = True
            raise

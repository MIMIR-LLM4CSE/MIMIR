"""Ray Serve LLM backend — the OpenAI-compatible router in front of vLLM engines.

Ray Serve LLM (``ray.serve.llm`` / ``build_openai_app``) exists to place, scale and
route model replicas across a GPU cluster; the engines it drives are vLLM, and what
it exposes is the same OpenAI API. So this backend is the OpenAI-compatible one pointed
at another address: the request builder, the reasoning profiles, the tool-call
normalisation and the streaming parser are the ones that already match those engines.

What genuinely differs is what the *router* serves, not what the engine does, and
:data:`RAY_ENDPOINT` declares it:

- its own address and key (``RAY_BASE_URL`` / ``RAY_API_KEY``);
- ``/tokenize`` is a vLLM-server extension the router does not have to proxy;
- ``max_model_len`` in ``/v1/models`` is likewise the vLLM server's field, so the
  window may have to be pinned by hand.

The ``ray`` package itself is not a dependency here — it runs on the cluster, and
this side only ever speaks HTTP.
"""
from __future__ import annotations

from ...config.models import RAY_API_KEY, RAY_BASE_URL
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

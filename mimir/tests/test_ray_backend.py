"""Tests for the Ray Serve backend — what it does *not* inherit from vLLM.

The chat path, the reasoning profiles and the tool-call handling are VllmBackend's
and are covered by its own tests. What is Ray's own is the endpoint it resolves,
and its behaviour when the router turns out not to serve vLLM's extra routes.
"""

import os
import unittest

from mimir.client.query_engine.backends import vllm_backend
from mimir.client.query_engine.backends.ray_backend import RayBackend, _get_ray_config


class _EnvGuard(unittest.TestCase):
    _KEYS = ("RAY_BASE_URL", "RAY_API_KEY", "VLLM_BASE_URL", "VLLM_API_KEY",
             "MIMIR_RAY_MAX_MODEL_LEN")

    def setUp(self) -> None:
        self._env = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class RayConfigTests(_EnvGuard):
    def test_v1_is_appended_once(self) -> None:
        os.environ["RAY_BASE_URL"] = "http://head:8000"
        self.assertEqual(_get_ray_config()[0], "http://head:8000/v1")

    def test_v1_the_user_typed_is_not_doubled(self) -> None:
        """A trailing slash is left as the user typed it — as on the vLLM side —
        but must not earn a second /v1."""
        os.environ["RAY_BASE_URL"] = "https://ray.internal/v1/"
        self.assertEqual(_get_ray_config()[0], "https://ray.internal/v1/")

    def test_route_prefix_is_preserved(self) -> None:
        """A Serve app mounted under a route prefix keeps it; /v1 goes after."""
        os.environ["RAY_BASE_URL"] = "http://head:8000/llm"
        self.assertEqual(_get_ray_config()[0], "http://head:8000/llm/v1")

    def test_vllm_variables_do_not_leak_in(self) -> None:
        """The whole point of the backend: its own address, not vLLM's."""
        os.environ["VLLM_BASE_URL"] = "http://vllm-box:8000"
        os.environ["VLLM_API_KEY"] = "vllm-secret"
        os.environ["RAY_BASE_URL"] = "http://ray-head:8000"
        base, key = RayBackend()._config()
        self.assertEqual(base, "http://ray-head:8000/v1")
        self.assertEqual(key, "EMPTY")

    def test_api_key_defaults_to_empty_sentinel(self) -> None:
        self.assertEqual(_get_ray_config()[1], "EMPTY")


class RayContextWindowTests(_EnvGuard):
    def test_env_override_wins_without_asking_the_router(self) -> None:
        os.environ["MIMIR_RAY_MAX_MODEL_LEN"] = "131072"
        calls = []
        original = vllm_backend.served_model_len
        vllm_backend.served_model_len = lambda *a, **k: calls.append(a) or 999
        try:
            self.assertEqual(RayBackend()._fetch_context_window("m"), 131_072)
        finally:
            vllm_backend.served_model_len = original
        self.assertEqual(calls, [])

    def test_non_positive_override_falls_through_to_the_router(self) -> None:
        os.environ["MIMIR_RAY_MAX_MODEL_LEN"] = "0"
        os.environ["RAY_BASE_URL"] = "http://head:8000"
        seen = []

        def _fake(model, config=None):
            seen.append((model, config))
            return 4096

        import mimir.client.query_engine.backends.ray_backend as rb
        original = rb.served_model_len
        rb.served_model_len = _fake
        try:
            self.assertEqual(RayBackend()._fetch_context_window("m"), 4096)
        finally:
            rb.served_model_len = original
        self.assertEqual(seen, [("m", ("http://head:8000/v1", "EMPTY"))])


class RayTokenizeTests(_EnvGuard):
    def test_first_failure_latches_so_later_counts_stay_off_the_network(self) -> None:
        """A router with no /tokenize must cost one round-trip, not one per count."""
        backend = RayBackend()
        attempts = []

        def _boom(self, model, text):
            attempts.append(text)
            raise RuntimeError("404 /tokenize")

        original = vllm_backend.VllmBackend._tokenize_text
        vllm_backend.VllmBackend._tokenize_text = _boom
        try:
            for text in ("first", "second", "third"):
                # count_text_tokens swallows the raise and uses the heuristic.
                self.assertGreater(backend.count_text_tokens("m", text), 0)
        finally:
            vllm_backend.VllmBackend._tokenize_text = original
        self.assertEqual(attempts, ["first"])
        self.assertTrue(backend._no_tokenize)

    def test_a_router_that_does_answer_is_used(self) -> None:
        backend = RayBackend()
        original = vllm_backend.VllmBackend._tokenize_text
        vllm_backend.VllmBackend._tokenize_text = lambda self, model, text: 7
        try:
            self.assertEqual(backend.count_text_tokens("m", "hello"), 7)
        finally:
            vllm_backend.VllmBackend._tokenize_text = original
        self.assertFalse(backend._no_tokenize)


class ServedModelsTests(_EnvGuard):
    def test_ray_enumerates_its_own_endpoint(self) -> None:
        os.environ["RAY_BASE_URL"] = "http://head:8000"
        seen = []
        original = vllm_backend._fetch_models
        vllm_backend._fetch_models = lambda config=None: seen.append(config) or [{"id": "Qwen3-32B"}]
        try:
            self.assertEqual(RayBackend().served_models(), ["Qwen3-32B"])
        finally:
            vllm_backend._fetch_models = original
        self.assertEqual(seen, [("http://head:8000/v1", "EMPTY")])

    def test_backends_without_a_listing_answer_empty(self) -> None:
        from mimir.client.query_engine.backends.ollama_backend import OllamaBackend
        self.assertEqual(OllamaBackend().served_models(), [])


class FactoryTests(_EnvGuard):
    def test_llm_backend_ray_resolves_to_the_ray_backend(self) -> None:
        from mimir.client.query_engine.backends.factory import (
            clear_backend_cache, get_backend,
        )
        previous = os.environ.get("LLM_BACKEND")
        os.environ["LLM_BACKEND"] = "ray"
        clear_backend_cache()
        try:
            self.assertIsInstance(get_backend(), RayBackend)
        finally:
            if previous is None:
                os.environ.pop("LLM_BACKEND", None)
            else:
                os.environ["LLM_BACKEND"] = previous
            clear_backend_cache()


if __name__ == "__main__":
    unittest.main()

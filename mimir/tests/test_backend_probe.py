"""The readiness probe that gates server startup.

Nothing else runs until ``_wait_for_backend`` returns: the WS server does not bind
its port, so a probe that waits on a verdict it already has costs the user the whole
timeout and shows them nothing. These tests pin the two halves of that: what counts
as ready, and what counts as never going to be ready.
"""
from __future__ import annotations

import asyncio
import queue

import httpx
import pytest

from mimir.client.ui.ws.ws_worker import _AgentWorker


class _Stub:
    """The only attribute the probe touches on ``self``."""

    def __init__(self) -> None:
        self.out_q: queue.Queue = queue.Queue()

    def messages(self) -> list[str]:
        out = []
        while not self.out_q.empty():
            out.append(self.out_q.get()["text"])
        return out


def _run(stub: _Stub) -> None:
    asyncio.run(_AgentWorker._wait_for_backend(stub))


@pytest.fixture
def vllm_env(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "vllm")
    monkeypatch.setenv("VLLM_BASE_URL", "https://endpoint.internal")
    # Any real wait would make a failing test hang rather than fail.
    monkeypatch.setenv("MIMIR_BACKEND_TIMEOUT", "0")
    monkeypatch.setenv("MIMIR_BACKEND_POLL_INTERVAL", "0")


def _responder(monkeypatch, handler):
    """Route every probe GET through *handler* (url -> Response, or raises)."""
    asked: list[str] = []

    def _get(self, url, *a, **kw):
        asked.append(str(url))
        return handler(str(url))

    monkeypatch.setattr(httpx.Client, "get", _get)
    return asked


class TestReady:
    def test_the_model_list_answering_is_what_ready_means(self, monkeypatch, vllm_env):
        asked = _responder(monkeypatch, lambda url: httpx.Response(200, json={"data": []}))
        _run(_Stub())
        # The request the agent depends on, not a side door that can differ from it.
        assert asked == ["https://endpoint.internal/v1/models"]

    def test_a_guarded_endpoint_is_up(self, monkeypatch, vllm_env):
        # 401/403 proves the server is answering; the agent's own request carries
        # the key. Treating it as "not ready" would wait out a live endpoint.
        _responder(monkeypatch, lambda url: httpx.Response(403))
        _run(_Stub())

    def test_a_base_url_already_ending_in_v1_is_not_doubled(self, monkeypatch, vllm_env):
        monkeypatch.setenv("VLLM_BASE_URL", "https://endpoint.internal/v1/")
        asked = _responder(monkeypatch, lambda url: httpx.Response(200, json={"data": []}))
        _run(_Stub())
        assert asked == ["https://endpoint.internal/v1/models"]

    def test_health_carries_a_vllm_still_loading_its_weights(self, monkeypatch, vllm_env):
        # A local vLLM answers /health before /v1/models. The fallback exists for
        # exactly that window, so a cold start is not reported as a dead endpoint.
        def handler(url):
            return httpx.Response(200) if url.endswith("/health") else httpx.Response(404)

        asked = _responder(monkeypatch, handler)
        _run(_Stub())
        assert asked == [
            "https://endpoint.internal/v1/models",
            "https://endpoint.internal/health",
        ]


class TestPermanentFailures:
    """Failures that answer the same on the hundredth attempt as on the first."""

    def test_a_refused_certificate_stops_the_wait_and_names_the_remedy(
        self, monkeypatch, vllm_env
    ):
        def handler(url):
            raise httpx.ConnectError(
                "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate"
            )

        _responder(monkeypatch, handler)
        stub = _Stub()
        with pytest.raises(RuntimeError) as exc:
            _run(stub)
        # Waiting cannot add a CA to the trust store, so the message must say what can.
        assert "VLLM_VERIFY_SSL" in str(exc.value)
        assert "did not become ready" not in str(exc.value)

    def test_nothing_served_anywhere_fails_fast(self, monkeypatch, vllm_env):
        _responder(monkeypatch, lambda url: httpx.Response(404))
        with pytest.raises(RuntimeError, match="404"):
            _run(_Stub())


class TestRetried:
    def test_a_server_not_up_yet_is_waited_on(self, monkeypatch, vllm_env):
        # A refused connection is the ordinary cold start: keep waiting, and fail
        # on the timeout rather than declaring the endpoint hopeless.
        def handler(url):
            raise httpx.ConnectError("connection refused")

        _responder(monkeypatch, handler)
        with pytest.raises(RuntimeError, match="did not become ready"):
            _run(_Stub())


class TestReporting:
    def test_the_attempt_is_announced_before_the_first_request(self, monkeypatch, vllm_env):
        _responder(monkeypatch, lambda url: httpx.Response(200, json={"data": []}))
        stub = _Stub()
        _run(stub)
        said = stub.messages()
        # During startup the WS port is not bound, so this also goes to stdout —
        # what the output channel shows while the user waits.
        assert "https://endpoint.internal/v1/models" in said[0]
        assert any("ready" in m for m in said)


class TestOtherBackends:
    def test_ollama_is_asked_for_its_tags(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        monkeypatch.setenv("MIMIR_BACKEND_TIMEOUT", "0")
        monkeypatch.setenv("MIMIR_BACKEND_POLL_INTERVAL", "0")
        asked = _responder(monkeypatch, lambda url: httpx.Response(200, json={"models": []}))
        _run(_Stub())
        assert asked == ["http://127.0.0.1:11434/api/tags"]

    def test_ray_is_asked_for_its_model_list_and_has_no_health_fallback(
        self, monkeypatch
    ):
        # /health belongs to the vLLM engine, not the Serve router: asking it behind
        # a route only spends a round-trip on a 404.
        monkeypatch.setenv("LLM_BACKEND", "ray")
        monkeypatch.setenv("RAY_BASE_URL", "http://head:8000/llm")
        monkeypatch.setenv("MIMIR_BACKEND_TIMEOUT", "0")
        monkeypatch.setenv("MIMIR_BACKEND_POLL_INTERVAL", "0")
        asked = _responder(monkeypatch, lambda url: httpx.Response(404))
        with pytest.raises(RuntimeError):
            _run(_Stub())
        assert asked == ["http://head:8000/llm/v1/models"]

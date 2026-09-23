"""Tests for the vLLM backend's request-shaping helpers and stop signal."""

import os
import types
import unittest
from unittest.mock import patch

from mimir.client.config.constants import CTX_RESERVED_RATIO
from mimir.client.query_engine.backends.vllm_backend import _answer_max_tokens


class AnswerMaxTokensTests(unittest.TestCase):
    """Regression: requested output must survive the estimate undercounting.

    Incident (2026-07-10): window 262144, local estimate ~16558 prompt tokens,
    vLLM counted 17071 (rendered chat template + tool schemas). The old
    ``mml - estimate - 512`` requested 245074 output tokens → 262145 total →
    400 "maximum context length".
    """

    def test_incident_numbers_stay_in_window(self) -> None:
        mml, estimate, actual = 262_144, 16_558, 17_071
        requested = _answer_max_tokens(mml, estimate)
        self.assertGreater(requested, 0)
        self.assertLessEqual(actual + requested, mml)

    def test_capped_at_answer_reserve(self) -> None:
        mml = 262_144
        self.assertEqual(_answer_max_tokens(mml, 16_558), int(mml * CTX_RESERVED_RATIO))

    def test_margin_scales_with_prompt(self) -> None:
        # A large prompt whose rendering overhead exceeds any fixed margin:
        # at 200K estimated tokens an eighth (25K) absorbs the undercount.
        mml, estimate = 262_144, 200_000
        requested = _answer_max_tokens(mml, estimate)
        undercount = estimate // 10  # generous real-world drift
        self.assertLessEqual(estimate + undercount + requested, mml)

    def test_near_window_returns_nonpositive_for_callsite_clamp(self) -> None:
        # The call site clamps to >= 1; the helper just must not blow up.
        self.assertLessEqual(_answer_max_tokens(262_144, 261_000), 0)


if __name__ == "__main__":
    unittest.main()


class FinishReasonTests(unittest.TestCase):
    """The stop signal rides on the *choice*, not the delta or the message.

    Streaming sends it only on the last chunk (null on every earlier one), so the
    reader has to keep the last non-null rather than whatever the final chunk held.
    """

    @staticmethod
    def _backend():
        from mimir.client.query_engine.backends.vllm_backend import VllmBackend
        b = VllmBackend()
        b._config = lambda: ("http://x/v1", "k")
        b._client_for = lambda *a, **k: object()
        return b

    @staticmethod
    def _chunk(finish=None, content=""):
        delta = types.SimpleNamespace(role="assistant", content=content, tool_calls=None)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason=finish, delta=delta)])

    def _run(self, streaming, response):
        import mimir.client.query_engine.backends.vllm_backend as vb
        with patch.object(vb, "_create", lambda client, kwargs: response), \
             patch.object(vb, "served_model_len", lambda model, config=None: None):
            return self._backend().chat(
                "m", [{"role": "user", "content": "q"}], [], False, streaming, {},
                token_callback=lambda t: None,
            )

    def test_non_streaming_reads_it_off_the_choice(self) -> None:
        message = types.SimpleNamespace(role="assistant", content="cut off",
                                        tool_calls=None)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason="length", message=message)])
        self.assertEqual(self._run(False, response)["finish_reason"], "length")

    def test_streaming_keeps_the_last_non_null(self) -> None:
        chunks = [self._chunk(None, "par"), self._chunk(None, "tial"),
                  self._chunk("length", "")]
        self.assertEqual(self._run(True, chunks)["finish_reason"], "length")

    def test_tool_calls_finish_is_normalized_too(self) -> None:
        message = types.SimpleNamespace(role="assistant", content="", tool_calls=None)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason="tool_calls", message=message)])
        self.assertEqual(self._run(False, response)["finish_reason"], "tool_calls")

    def test_a_provider_that_says_nothing_adds_no_key(self) -> None:
        message = types.SimpleNamespace(role="assistant", content="hi", tool_calls=None)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason=None, message=message)])
        self.assertNotIn("finish_reason", self._run(False, response))


class SamplingParamsTests(unittest.TestCase):
    """The model's generation_config decides sampling unless a caller overrides it.

    A forced low temperature sent a reasoning model into repetition loops, so the
    backend adds no default of its own.
    """

    def _sent(self, options):
        import mimir.client.query_engine.backends.vllm_backend as vb
        sent = {}
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason="stop", message=message)])

        def _create(client, kwargs):
            sent.update(kwargs)
            return response

        with patch.object(vb, "_create", _create), \
             patch.object(vb, "served_model_len", lambda model, config=None: None):
            FinishReasonTests._backend().chat(
                "m", [{"role": "user", "content": "q"}], [], False, False, options)
        return sent

    def test_no_temperature_unless_asked(self) -> None:
        self.assertNotIn("temperature", self._sent({}))

    def test_an_explicit_temperature_is_forwarded(self) -> None:
        self.assertEqual(self._sent({"temperature": 0.0})["temperature"], 0.0)


class TokenizeAbsenceTests(unittest.TestCase):
    """Regression: a router with no /tokenize cost a round-trip per message per pass.

    Incident (2026-09-18): an OpenAI-compatible router answered /tokenize with 500
    "no valid backends". The heuristic fallback was not cached, so every recount of
    the history went back to the network and turns waited 25-76 s before the model
    was even asked.
    """

    def _backend(self, status: int):
        import httpx
        from mimir.client.query_engine.backends.vllm_backend import VllmBackend

        calls = []

        def _handler(request):
            calls.append(request.url.path)
            if status == 200:
                return httpx.Response(200, json={"count": 7})
            return httpx.Response(status, text="no valid backends")

        backend = VllmBackend()
        backend._tokenize_http = httpx.Client(transport=httpx.MockTransport(_handler))
        return backend, calls

    def _count(self, backend, texts):
        with patch.dict("os.environ", {"VLLM_BASE_URL": "http://router:8000"}):
            return [backend.count_text_tokens("m", t) for t in texts]

    def test_a_definitive_refusal_is_asked_once(self) -> None:
        for status in (404, 405, 500, 501):
            with self.subTest(status=status):
                backend, calls = self._backend(status)
                counts = self._count(backend, ["first", "second", "third"])
                self.assertTrue(all(c > 0 for c in counts))
                self.assertEqual(calls, ["/tokenize"])

    def test_a_busy_server_is_asked_again(self) -> None:
        backend, calls = self._backend(503)
        self._count(backend, ["first", "second"])
        self.assertEqual(len(calls), 2)

    def test_a_server_that_answers_is_used(self) -> None:
        backend, calls = self._backend(200)
        self.assertEqual(self._count(backend, ["hello"]), [7])

    def test_the_refusal_belongs_to_its_endpoint(self) -> None:
        backend, calls = self._backend(500)
        self._count(backend, ["first"])
        with patch.dict("os.environ", {"VLLM_BASE_URL": "http://other:8000"}):
            backend.count_text_tokens("m", "second")
        self.assertEqual(len(calls), 2)

    def test_the_switch_keeps_every_count_off_the_network(self) -> None:
        # A router that times out is retried on every count: the switch is the only
        # thing that keeps it from costing seconds, even on the first call.
        for value in ("0", "false", "off"):
            with self.subTest(value=value):
                backend, calls = self._backend(200)
                with patch.dict("os.environ", {"MIMIR_VLLM_TOKENIZE": value}):
                    counts = self._count(backend, ["first", "second"])
                self.assertTrue(all(c > 0 for c in counts))
                self.assertEqual(calls, [])


class ContextWindowPriorityTests(unittest.TestCase):
    """Priority: the server's reported window, then the override, then unknown.

    The reported ``max_model_len`` is authoritative; ``MIMIR_VLLM_MAX_MODEL_LEN``
    is the fallback for servers whose /v1/models reports none, and None lets the
    caller keep its static budget when neither is available.
    """

    _ENV = ("MIMIR_VLLM_MAX_MODEL_LEN",)

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in self._ENV}
        for k in self._ENV:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _backend(self, reported):
        import mimir.client.query_engine.backends.vllm_backend as vb
        backend = vb.VllmBackend()
        backend._config = lambda: ("http://x/v1", "k")
        backend._saved_report = vb.served_model_len
        vb.served_model_len = lambda model, config=None: reported
        return vb, backend

    def test_reported_window_wins_over_the_override(self) -> None:
        os.environ["MIMIR_VLLM_MAX_MODEL_LEN"] = "512000"
        vb, backend = self._backend(32768)
        try:
            self.assertEqual(backend._fetch_context_window("m"), 32_768)
        finally:
            vb.served_model_len = backend._saved_report

    def test_override_is_used_when_the_server_reports_nothing(self) -> None:
        os.environ["MIMIR_VLLM_MAX_MODEL_LEN"] = "512000"
        vb, backend = self._backend(None)
        try:
            self.assertEqual(backend._fetch_context_window("m"), 512_000)
        finally:
            vb.served_model_len = backend._saved_report

    def test_neither_reported_nor_overridden_stays_unknown(self) -> None:
        vb, backend = self._backend(None)
        try:
            self.assertIsNone(backend._fetch_context_window("m"))
        finally:
            vb.served_model_len = backend._saved_report



class UnknownWindowTests(unittest.TestCase):
    """A server that publishes no max_model_len must still bound the answer.

    Incident (2026-09-20): an OpenAI-compatible router served the model but
    reported no ``max_model_len``, so no ``max_tokens`` was sent and vLLM defaulted
    it to the whole remaining window — a >600K-token allowance. A sub-agent's
    report-writing step ran until the router gave up with a 500, three identical
    retries deep, ~90s apiece, with nothing to show for the six minutes.
    """

    def _sent(self, options, window=None):
        import mimir.client.query_engine.backends.vllm_backend as vb
        sent: dict = {}
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason="stop", message=message)])

        def _create(client, kwargs):
            sent.update(kwargs)
            return response

        with patch.object(vb, "_create", _create), \
             patch.object(vb, "served_model_len", lambda model, config=None: window):
            FinishReasonTests._backend().chat(
                "m", [{"role": "user", "content": "q"}], [], False, False, options)
        return sent

    def test_no_published_window_still_caps_the_answer(self) -> None:
        from mimir.client.config.constants import CTX_TOTAL_FULL
        sent = self._sent({})
        self.assertEqual(sent["max_tokens"], int(CTX_TOTAL_FULL * CTX_RESERVED_RATIO))

    def test_an_explicit_ceiling_still_wins(self) -> None:
        self.assertEqual(self._sent({"max_tokens": 128})["max_tokens"], 128)

    def test_a_published_window_sizes_the_answer_as_before(self) -> None:
        sent = self._sent({}, window=262_144)
        self.assertEqual(sent["max_tokens"], int(262_144 * CTX_RESERVED_RATIO))


class DeclaredWindowBoundsTheAnswerTests(unittest.TestCase):
    """The window that bounds generation is the one every other budget uses.

    Incident (2026-09-22): a router publishing no ``max_model_len``, with the window
    declared as 512K in the settings. ``_fetch_context_window`` honoured the
    declaration, so the context bar, the eviction and the compaction all sized
    themselves to 512K — but the request shaper read ``served_model_len`` directly,
    which consults only /v1/models. The declaration was invisible at the one place
    that caps a generation, so a README-writing step fell through to the
    unknown-window reserve, ran for minutes, and died on the router's 500.
    """

    def _sent(self, options, declared=None, served=None):
        import mimir.client.query_engine.backends.vllm_backend as vb
        sent: dict = {}
        message = types.SimpleNamespace(role="assistant", content="ok", tool_calls=None)
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(finish_reason="stop", message=message)])
        env = {"MIMIR_VLLM_MAX_MODEL_LEN": str(declared)} if declared is not None else {}

        def _create(client, kwargs):
            sent.update(kwargs)
            return response

        with patch.dict("os.environ", env, clear=False), \
             patch.object(vb, "_create", _create), \
             patch.object(vb, "served_model_len", lambda model, config=None: served):
            if declared is None:
                os.environ.pop("MIMIR_VLLM_MAX_MODEL_LEN", None)
            FinishReasonTests._backend().chat(
                "m", [{"role": "user", "content": "q"}], [], False, False, options)
        return sent

    def test_a_declared_window_sizes_the_answer_when_none_is_published(self) -> None:
        sent = self._sent({}, declared=524_288)
        self.assertEqual(sent["max_tokens"], int(524_288 * CTX_RESERVED_RATIO))

    def test_a_callers_ceiling_is_clamped_to_what_the_window_allows(self) -> None:
        # A caller sizes its ceiling from the work, not from this endpoint; sent
        # verbatim, a 200K allocation on a 32K model is a 400 rather than a cap.
        sent = self._sent({"max_tokens": 200_000}, served=32_768)
        self.assertEqual(sent["max_tokens"], int(32_768 * CTX_RESERVED_RATIO))

    def test_a_ceiling_under_the_window_is_left_alone(self) -> None:
        self.assertEqual(self._sent({"max_tokens": 4_096}, served=262_144)["max_tokens"],
                         4_096)

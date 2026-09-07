"""Tests for the vLLM backend's request-shaping helpers and stop signal."""

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

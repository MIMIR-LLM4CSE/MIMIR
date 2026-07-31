"""Tests for the vLLM backend's request-shaping helpers."""

import unittest

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

"""The stop signal is normalized once, and never travels back to the provider.

Every provider spells it differently — OpenAI/vLLM ``finish_reason``, Ollama
``done_reason``, Anthropic ``stop_reason`` — and all three used to drop it, which
left a turn cut off at ``max_tokens`` indistinguishable from a model with nothing
to say. Both surfaced as the same "empty turn" retry.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from mimir.client.query_engine.backends.base import normalize_finish_reason
from mimir.client.query_engine.streaming import _process_response, _note_truncated_turn


class NormalizeFinishReasonTests(unittest.TestCase):
    def test_openai_vocabulary(self) -> None:
        for raw, want in [("stop", "stop"), ("length", "length"),
                          ("tool_calls", "tool_calls"), ("content_filter", "content_filter")]:
            self.assertEqual(normalize_finish_reason(raw), want)

    def test_anthropic_vocabulary(self) -> None:
        for raw, want in [("end_turn", "stop"), ("stop_sequence", "stop"),
                          ("max_tokens", "length"), ("tool_use", "tool_calls"),
                          ("refusal", "content_filter")]:
            self.assertEqual(normalize_finish_reason(raw), want)

    def test_absent_is_not_unknown(self) -> None:
        """Nothing said and something unrecognized are different facts."""
        self.assertIsNone(normalize_finish_reason(None))
        self.assertIsNone(normalize_finish_reason(""))
        self.assertEqual(normalize_finish_reason("brand_new_reason"), "unknown")

    def test_case_and_whitespace_are_not_significant(self) -> None:
        self.assertEqual(normalize_finish_reason("  MAX_TOKENS "), "length")


class HistoryLeakTests(unittest.TestCase):
    """finish_reason describes the call, not the message."""

    def test_it_is_stripped_before_the_message_is_appended(self) -> None:
        messages: list[dict] = []
        msg = {"role": "assistant", "content": "hi",
               "thinking": "…", "finish_reason": "length"}
        with patch("mimir.client.query_engine.streaming.emit"):
            _process_response(msg, messages, thinking=True)
        self.assertEqual(len(messages), 1)
        self.assertNotIn("finish_reason", messages[0])
        self.assertNotIn("thinking", messages[0])
        self.assertEqual(messages[0]["content"], "hi")

    def test_the_caller_can_still_read_it_off_the_original(self) -> None:
        msg = {"role": "assistant", "content": "hi", "finish_reason": "length"}
        with patch("mimir.client.query_engine.streaming.emit"):
            _process_response(msg, [], thinking=False)
        self.assertEqual(msg["finish_reason"], "length")


class TruncationNoticeTests(unittest.TestCase):
    def test_a_truncated_turn_is_announced(self) -> None:
        emitted: list[dict] = []
        with patch("mimir.client.query_engine.streaming.emit", emitted.append):
            _note_truncated_turn({"finish_reason": "length"}, step=7)
        self.assertEqual(len(emitted), 1)
        self.assertIn("finish_reason=length", emitted[0]["text"])
        self.assertIn("step 7", emitted[0]["text"])

    def test_a_normal_turn_says_nothing(self) -> None:
        emitted: list[dict] = []
        with patch("mimir.client.query_engine.streaming.emit", emitted.append):
            _note_truncated_turn({"finish_reason": "stop"})
            _note_truncated_turn({})
        self.assertEqual(emitted, [])


class ScriptedBackendTests(unittest.TestCase):
    def test_a_scripted_turn_can_carry_a_finish_reason(self) -> None:
        from mimir.tests._fake_backend import ScriptedBackend

        backend = ScriptedBackend([{"content": "cut off", "finish_reason": "length"}])
        out = backend.chat("m", [], [], False, False, {})
        self.assertEqual(out["finish_reason"], "length")


if __name__ == "__main__":
    unittest.main()

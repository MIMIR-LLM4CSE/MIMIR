"""Only the agent loop's own calls calibrate the context bar's overhead.

Regression cover for a bar that swung back and forth mid-session: every call on the
shared backend stored its reported prompt size as the model's overhead, including
the skill classifier and the session summary, whose prompt is a short instruction
and no tools. The overhead fell to almost nothing after each of them and came back
on the next agent step.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from mimir.client.query_engine import streaming
from mimir.client.query_engine.backends.base import LLMBackend


class _Backend(LLMBackend):
    """Reports a fixed prompt size and counts every message as ten tokens."""

    def __init__(self, reported: int) -> None:
        super().__init__()
        self.reported = reported

    def chat(self, model, messages, tools, thinking, streaming, options, **_):
        return {"role": "assistant", "content": "ok", "prompt_tokens": self.reported}

    def count_messages_tokens(self, model, messages, allow_network=True):
        return 10 * len(messages)

    def count_text_tokens(self, model, text, allow_network=True):
        return len(text)

    def context_window(self, model):
        return 1000

    def list_models(self):
        return []


_HISTORY = [
    {"role": "system", "content": "doctrine"},
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
]


class OverheadCalibrationTests(unittest.TestCase):

    def test_agent_step_calibrates_and_strips_the_figure(self) -> None:
        backend = _Backend(reported=5020)
        with patch.object(streaming, "get_backend", return_value=backend):
            msg = streaming._stream_chat("m", _HISTORY, [], False, False, {})
        self.assertNotIn("prompt_tokens", msg)
        self.assertEqual(backend.measured_prompt_overhead("m"), 5000)

    def test_side_call_leaves_the_overhead_alone(self) -> None:
        backend = _Backend(reported=5020)
        with patch.object(streaming, "get_backend", return_value=backend):
            streaming._stream_chat("m", _HISTORY, [], False, False, {})
        backend.reported = 300
        backend.chat("m", [{"role": "system", "content": "classify"}], [], False, False, {})
        self.assertEqual(backend.measured_prompt_overhead("m"), 5000)


if __name__ == "__main__":
    unittest.main()

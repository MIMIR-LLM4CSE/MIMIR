"""Only the agent loop's own calls calibrate the context bar's overhead.

Regression cover for a bar that swung back and forth mid-session: every call on the
shared backend stored its reported prompt size as the model's overhead, including
the skill classifier and the session summary, whose prompt is a short instruction
and no tools. The overhead fell to almost nothing after each of them and came back
on the next agent step.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mimir.client.query_engine import streaming
from mimir.client.query_engine.backends.base import LLMBackend, message_wire_form


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


_TOOLS = [{"type": "function", "function": {"name": "t", "description": "d" * 140_000}}]


def _fixed_tokens(messages: list[dict], tools: list[dict]) -> int:
    system = messages[0] if messages and messages[0].get("role") == "system" else None
    chars = (len(message_wire_form(system)) if system else 0) + (len(json.dumps(tools)) if tools else 0)
    return int(chars / 4)


def _history_tokens(history: list[dict]) -> int:
    """What the server charges for *history*: dense stretches at 3, the rest at 5."""
    return sum(
        int(len(message_wire_form(m)) / (3 if "dense" in m["content"] else 5))
        for m in history
    )


class _Server(LLMBackend):
    """No tokenizer. Charges the fixed part at the heuristic's own 4 chars/token and
    the history at a density that changes from one stretch to the next: the shape
    of a real DeepSeek session, where the ratio ran from 3.3 to 4.2."""

    def chat(self, model, messages, tools, thinking, streaming, options, **_):
        return {"role": "assistant", "content": "ok",
                "prompt_tokens": _fixed_tokens(messages, tools) + _history_tokens(messages[1:])}

    def list_models(self):
        return []


def _history(n_turns: int, dense_turns: int = 0) -> list[dict]:
    msgs = [{"role": "system", "content": "doctrine " * 1000}]
    for i in range(n_turns):
        kind = "dense" if i < dense_turns else "prose"
        msgs.append({"role": "user", "content": f"{kind} {i} " + "x" * 8000})
        msgs.append({"role": "assistant", "content": f"{kind} " + "y" * 8000})
    return msgs


class CharsPerTokenCalibrationTests(unittest.TestCase):
    """The history's ratio is measured from the server, with no /tokenize at all."""

    def _step(self, backend, messages, tools=_TOOLS):
        with patch.object(streaming, "get_backend", return_value=backend):
            streaming._stream_chat("m", messages, tools, False, False, {})

    def test_one_step_measures_the_history_and_the_true_overhead(self) -> None:
        backend = _Server()
        messages = _history(6)
        self._step(backend, messages)
        self.assertAlmostEqual(backend.calibrated_chars_per_token("m"), 5.0, delta=0.05)
        self.assertAlmostEqual(backend.measured_prompt_overhead("m"),
                               _fixed_tokens(messages, _TOOLS), delta=50)

    def test_a_mixed_history_is_counted_at_its_average(self) -> None:
        # A ratio taken from the difference between two calls measures only the last
        # stretch; the whole history's average is what the whole history costs.
        backend = _Server()
        messages = _history(8, dense_turns=4)
        self._step(backend, messages)
        counted = backend.count_messages_tokens("m", messages[1:], allow_network=False)
        self.assertAlmostEqual(counted, _history_tokens(messages[1:]), delta=50)

    def test_a_short_history_is_not_a_measurement(self) -> None:
        backend = _Server()
        self._step(backend, _history(0) + [{"role": "user", "content": "hi"}])
        self.assertIsNone(backend.calibrated_chars_per_token("m"))

    def test_an_implausible_ratio_is_ignored(self) -> None:
        backend = _Server()
        messages = _history(4)
        # Far more tokens than the history's characters could make: not a tokenizer.
        backend.note_prompt_usage("m", messages, _fixed_tokens(messages, _TOOLS) + 60_000,
                                  tools=_TOOLS)
        self.assertIsNone(backend.calibrated_chars_per_token("m"))

    def test_the_calibrated_ratio_drives_the_heuristic(self) -> None:
        backend = _Server()
        text = "z" * 10_000
        self.assertEqual(backend.count_text_tokens("m", text), 2500)
        self._step(backend, _history(6))
        # Not cached from before the calibration: the next count uses the new ratio.
        self.assertAlmostEqual(backend.count_text_tokens("m", text), 2000, delta=20)

    def test_a_side_call_does_not_calibrate(self) -> None:
        backend = _Server()
        backend.chat("m", _history(6), [], False, False, {})
        self.assertIsNone(backend.calibrated_chars_per_token("m"))


if __name__ == "__main__":
    unittest.main()

"""Tests for the Ollama backend's chat path.

``done_reason`` sits on the response envelope, never on ``message`` — the reader
that goes looking for it inside ``message`` finds nothing and reports no stop
signal at all, which is indistinguishable from a provider that sent none.
"""

import unittest
from unittest.mock import patch

import mimir.client.query_engine.backends.ollama_backend as ob
from mimir.client.query_engine.backends.ollama_backend import OllamaBackend


def _chunk(content="", done_reason=None):
    chunk = {"message": {"role": "assistant", "content": content}}
    if done_reason is not None:
        chunk["done_reason"] = done_reason
    return chunk


class DoneReasonTests(unittest.TestCase):
    def _run(self, streaming, answer):
        with patch.object(ob.ollama, "chat", lambda **kw: answer):
            return OllamaBackend().chat(
                "m", [{"role": "user", "content": "q"}], [], False, streaming,
                {"num_ctx": 4096}, token_callback=lambda t: None,
            )

    def test_non_streaming_reads_the_envelope(self) -> None:
        out = self._run(False, _chunk("cut off", done_reason="length"))
        self.assertEqual(out["finish_reason"], "length")
        self.assertEqual(out["content"], "cut off")

    def test_streaming_reads_the_terminal_chunk(self) -> None:
        chunks = [_chunk("par"), _chunk("tial"), _chunk("", done_reason="length")]
        self.assertEqual(self._run(True, chunks)["finish_reason"], "length")

    def test_a_normal_stop_is_normalized(self) -> None:
        self.assertEqual(self._run(False, _chunk("done", done_reason="stop"))["finish_reason"],
                         "stop")

    def test_an_envelope_without_the_field_adds_no_key(self) -> None:
        self.assertNotIn("finish_reason", self._run(False, _chunk("hi")))

    def test_a_done_reason_inside_message_is_not_mistaken_for_the_envelope(self) -> None:
        """Guard against reading it off the wrong object and reporting a false stop."""
        answer = {"message": {"role": "assistant", "content": "hi",
                              "done_reason": "length"}}
        self.assertNotIn("finish_reason", self._run(False, answer))


if __name__ == "__main__":
    unittest.main()

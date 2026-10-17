"""What every backend's call produces, assembled in one place (no network)."""

import asyncio
import threading
import unittest

from mimir.client.query_engine.backends.base import ChatTurn


class _Events:
    def __init__(self) -> None:
        self.seen: list[tuple[str, str]] = []

    def callbacks(self) -> dict:
        return {
            "token_callback": lambda t: self.seen.append(("token", t)),
            "think_token_callback": lambda t: self.seen.append(("think", t)),
            "think_start_callback": lambda: self.seen.append(("start", "")),
            "think_end_callback": lambda: self.seen.append(("end", "")),
        }


class ChatTurnTests(unittest.TestCase):
    def test_streaming_routes_reasoning_and_prose_to_the_callbacks(self) -> None:
        ev = _Events()
        turn = ChatTurn(streaming=True, **ev.callbacks())
        turn.reasoning("why")
        turn.content("answer")
        turn.stopped("max_tokens")
        msg = turn.result()
        self.assertEqual(msg["content"], "answer")
        self.assertEqual(msg["thinking"], "why")
        self.assertEqual(msg["finish_reason"], "length")
        self.assertIn(("think", "why"), ev.seen)
        self.assertEqual(ev.seen[-1], ("end", ""))

    def test_without_streaming_reasoning_never_reaches_the_callbacks(self) -> None:
        # process_response emits the whole block in that mode; a tag-based model's
        # thinking used to go out through the callbacks as well, and showed twice.
        ev = _Events()
        turn = ChatTurn(streaming=False, **ev.callbacks())
        turn.content("<think>why</think>answer")
        msg = turn.result()
        self.assertEqual(msg["thinking"], "why")
        self.assertEqual(ev.seen, [("token", "answer")])

    def test_stop_is_honoured(self) -> None:
        flag = threading.Event()
        flag.set()
        with self.assertRaises(asyncio.CancelledError):
            ChatTurn(streaming=True, cancel_flag=flag).check_cancelled()


class ReadinessTests(unittest.TestCase):
    def test_each_backend_says_what_to_wait_on(self) -> None:
        import os
        from unittest import mock

        from mimir.client.query_engine.backends.anthropic_backend import AnthropicBackend
        from mimir.client.query_engine.backends.ray_backend import RayBackend
        from mimir.client.query_engine.backends.vllm_backend import VllmBackend

        self.assertEqual(AnthropicBackend().readiness_urls(), [])
        with mock.patch.dict(os.environ, {"VLLM_BASE_URL": "http://n:8000",
                                          "RAY_BASE_URL": "http://h:8000/llm"}):
            self.assertEqual(VllmBackend().readiness_urls(),
                             ["http://n:8000/v1/models", "http://n:8000/health"])
            self.assertEqual(RayBackend().readiness_urls(), ["http://h:8000/llm/v1/models"])


if __name__ == "__main__":
    unittest.main()

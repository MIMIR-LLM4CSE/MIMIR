"""What a model call reports of itself, and what the history's fitting says it did.

The usage (prompt, completion, prefix-cached tokens) rides on the message out of the
backend and never into history; a side call (a compaction) is told from the loop's own;
each pass that shrinks the context says so in a ``status`` event carrying ``context``.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from mimir.client.llm import openai_compat
from mimir.client.llm.base import CALL_USAGE_KEYS, ChatTurn, call_purpose, side_call
from mimir.client.protocol.event_sink import event_sink
from mimir.client.query_engine import history as history_module
from mimir.client.query_engine.streaming import process_response
from mimir.tests.support.context import build_execution_context


class _Details:
    def __init__(self, cached: int) -> None:
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt: int, completion: int, cached: int | None) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached) if cached is not None else None


class CallUsageTests(unittest.TestCase):
    def test_the_usage_block_reaches_the_message(self) -> None:
        turn = ChatTurn(streaming=False, token_callback=lambda _t: None)
        openai_compat._record_usage(turn, _Usage(1200, 80, 1000))
        msg = turn.result()
        self.assertEqual({k: msg[k] for k in CALL_USAGE_KEYS},
                         {"prompt_tokens": 1200, "completion_tokens": 80, "cached_tokens": 1000})

    def test_no_usage_block_reports_nothing(self) -> None:
        turn = ChatTurn(streaming=False, token_callback=lambda _t: None)
        openai_compat._record_usage(turn, None)
        openai_compat._record_usage(turn, _Usage(0, 0, None))
        self.assertFalse(set(CALL_USAGE_KEYS) & set(turn.result()))

    def test_the_usage_never_enters_history(self) -> None:
        messages: list[dict] = []
        process_response({"role": "assistant", "content": "ok", "prompt_tokens": 5,
                          "completion_tokens": 2, "cached_tokens": 1}, messages, False)
        self.assertEqual(messages, [{"role": "assistant", "content": "ok"}])

    def test_a_side_call_is_named_and_the_loop_resumes_after_it(self) -> None:
        self.assertEqual(call_purpose(), "loop")
        with side_call("compaction"):
            self.assertEqual(call_purpose(), "compaction")
        self.assertEqual(call_purpose(), "loop")


def _conversation(n: int) -> list[dict]:
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "task"}]
    for i in range(n):
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "read", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 400})
    return messages


class ContextEventTests(unittest.TestCase):
    def _events(self, run) -> list[dict]:
        seen: list[dict] = []
        with event_sink(seen.append):
            run()
        return [e["context"] for e in seen if isinstance(e.get("context"), dict)]

    def test_a_compaction_says_what_it_brought_the_context_to(self) -> None:
        messages = _conversation(8)
        summary = [{"role": "assistant", "content": "[Context summary] done"}]
        events = self._events(lambda: history_module._maybe_compact_intra_query(
            messages, "S", build_execution_context(), lambda middle: summary,
            token_counter=len, token_budget=100))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "compact")
        self.assertTrue(events[0]["summarised"])
        self.assertLess(events[0]["after"], events[0]["before"])

    def test_a_failed_summary_says_so_and_leaves_the_history_whole(self) -> None:
        messages = _conversation(8)
        before = [dict(m) for m in messages]
        events = self._events(lambda: history_module._maybe_compact_intra_query(
            messages, "S", build_execution_context(), lambda middle: middle,
            token_counter=len, token_budget=100))
        self.assertEqual(events, [{"kind": "compact", "before": events[0]["before"],
                                   "after": events[0]["before"], "summarised": False}])
        self.assertEqual(messages, before)

    def test_an_eviction_counts_what_it_dropped(self) -> None:
        messages = _conversation(6)
        events = self._events(lambda: history_module._trim_tool_history(
            messages, execution_context=build_execution_context(), token_counter=len,
            token_budget=1000))
        self.assertEqual(events[0]["kind"], "trim")
        self.assertGreater(events[0]["evicted"], 0)
        self.assertGreater(events[0]["freed"], 0)

    def test_nothing_to_evict_says_nothing(self) -> None:
        with patch.object(history_module, "emit") as emitted:
            history_module._trim_tool_history(_conversation(1), token_counter=len,
                                              token_budget=10_000)
        emitted.assert_not_called()


if __name__ == "__main__":
    unittest.main()

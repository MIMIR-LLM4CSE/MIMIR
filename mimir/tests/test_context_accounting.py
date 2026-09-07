"""The prompt is measured the way the provider measures it.

Regression cover for the accounting gap behind session ``b2fd38f1``: the trim and
compaction triggers sized history with ``content`` alone, while the request the
backend serialized carried ``tool_calls[].function.arguments`` too. On that run the
budgeteer saw ~44k tokens where the prompt held ~80k — of which 36k were tool-call
arguments, 22k of those ``write_file`` bodies — so neither trigger ever fired and
the force-fit backstop, which *did* count the arguments, had no way to reduce them.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from mimir.client.query_engine import history as history_module
from mimir.client.query_engine.history import (
    _digest_call_args,
    _digest_failed_call_args,
    _force_fit_to_window,
    _maybe_compact_intra_query,
    _message_tokens,
    reconcile_tool_pairs,
)


def _call(call_id: str, name: str = "write_file", **args) -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": dict(args)}}


def _assistant(*calls: dict, content=None) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": list(calls)}


def _ok(call_id: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps({"status": "ok", "operation": "updated"})}


def _err(call_id: str, error: str = "Write policy blocked tool 'write_file'.") -> dict:
    return {"role": "tool", "tool_call_id": call_id,
            "content": json.dumps({"status": "error", "error": error})}


def _content_only(messages: list[dict]) -> int:
    """The old, blind measure: content and nothing else."""
    return sum(len(m.get("content") or "") for m in messages)


class MessageTokensTests(unittest.TestCase):
    def test_counts_content_and_tool_call_arguments(self) -> None:
        m = _assistant(_call("c1", path="a.py", content="x" * 100), content="prose")
        self.assertGreater(_message_tokens(m, len), 100)

    def test_tolerates_content_none(self) -> None:
        """An assistant turn that only calls tools has ``content`` present but None.

        Both former call sites passed that None straight to the token counter.
        """
        m = _assistant(_call("c1", path="a.py"))
        self.assertIsNone(m["content"])
        self.assertGreater(_message_tokens(m, len), 0)

    def test_empty_message_costs_nothing(self) -> None:
        self.assertEqual(_message_tokens({"role": "user", "content": ""}, len), 0)


class CompactionTriggerTests(unittest.TestCase):
    """The regression from ``b2fd38f1``, reduced to a unit."""

    @staticmethod
    def _history() -> list[dict]:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        for i in range(4):
            messages.append(_assistant(_call(f"c{i}", path=f"f{i}.py", content="x" * 400)))
            messages.append(_ok(f"c{i}"))
        return messages

    def test_weight_in_arguments_is_invisible_to_the_content_only_measure(self) -> None:
        messages = self._history()
        budget = 1_000
        # The premise: under the old metric this history reads as comfortably small…
        self.assertLess(_content_only(messages), budget)
        # …while what actually goes on the wire is well over the budget.
        self.assertGreater(sum(_message_tokens(m, len) for m in messages), budget)

    def test_compaction_fires_on_a_history_whose_weight_is_in_arguments(self) -> None:
        messages = self._history()
        original = len(messages)

        def compact_fn(middle: list[dict]) -> list[dict]:
            return [{"role": "assistant", "content": "[summary]"}]

        with patch.object(history_module, "emit"):
            _maybe_compact_intra_query(
                messages, "sys", {}, compact_fn, token_counter=len, token_budget=1_000,
            )
        self.assertLess(len(messages), original)
        self.assertEqual(messages[2]["content"], "[summary]")

    def test_a_genuinely_small_history_is_still_left_alone(self) -> None:
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        for i in range(4):
            messages.append(_assistant(_call(f"c{i}", path=f"f{i}.py")))
            messages.append(_ok(f"c{i}"))
        before = list(messages)
        _maybe_compact_intra_query(
            messages, "sys", {}, lambda middle: [{"role": "assistant", "content": "x"}],
            token_counter=len, token_budget=100_000,
        )
        self.assertEqual(messages, before)


class ForceFitArgumentsTests(unittest.TestCase):
    """The backstop must be able to reduce what it counts."""

    @staticmethod
    def _history() -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            _assistant(_call("old", path="big.py", content="x" * 4_000)),
            _ok("old"),
            _assistant(_call("new", path="small.py", content="y" * 10)),
            _ok("new"),
            {"role": "user", "content": "question"},
        ]

    def test_a_message_with_no_content_but_huge_arguments_is_reducible(self) -> None:
        messages = self._history()
        self.assertIsNone(messages[1]["content"])  # nothing for the old pass to shrink
        fitted = _force_fit_to_window(messages, 600, len)
        self.assertTrue(fitted)
        self.assertLessEqual(sum(_message_tokens(m, len) for m in messages), 600)
        self.assertIn("elided", messages[1]["tool_calls"][0]["function"]["arguments"]["content"])

    def test_the_newest_call_turn_keeps_its_arguments(self) -> None:
        """The turn whose results are being answered must stay legible to the model."""
        messages = self._history()
        _force_fit_to_window(messages, 600, len)
        self.assertEqual(messages[3]["tool_calls"][0]["function"]["arguments"]["content"],
                         "y" * 10)

    def test_still_reports_failure_when_the_core_alone_is_too_big(self) -> None:
        messages = [
            {"role": "system", "content": "s" * 5_000},
            {"role": "user", "content": "u" * 5_000},
        ]
        self.assertFalse(_force_fit_to_window(messages, 100, len))


class FailedCallDigestTests(unittest.TestCase):
    """Arguments of a call that never took effect are dead weight, not context."""

    @staticmethod
    def _history() -> list[dict]:
        return [
            {"role": "system", "content": "sys"},
            _assistant(_call("rejected", path="snap.py", content="x" * 2_000)),
            _err("rejected"),
            _assistant(_call("retry", path="snap.py", content="y" * 2_000)),
            _err("retry", "Syntax error in resulting Python content — file NOT written."),
            {"role": "user", "content": "q"},
        ]

    def test_an_older_failed_call_is_digested(self) -> None:
        messages = self._history()
        self.assertEqual(_digest_failed_call_args(messages), 1)
        args = messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIn("elided", args["content"])
        self.assertIn("2000 chars", args["content"])

    def test_the_newest_failure_keeps_its_arguments(self) -> None:
        """The model repairing a syntax error needs to see what it just wrote."""
        messages = self._history()
        _digest_failed_call_args(messages)
        self.assertEqual(messages[3]["tool_calls"][0]["function"]["arguments"]["content"],
                         "y" * 2_000)

    def test_a_successful_call_is_never_digested(self) -> None:
        messages = self._history()
        messages[2] = _ok("rejected")
        self.assertEqual(_digest_failed_call_args(messages), 0)
        self.assertEqual(messages[1]["tool_calls"][0]["function"]["arguments"]["content"],
                         "x" * 2_000)

    def test_digesting_twice_is_stable(self) -> None:
        messages = self._history()
        self.assertEqual(_digest_failed_call_args(messages), 1)
        self.assertEqual(_digest_failed_call_args(messages), 0)

    def test_scalar_arguments_survive_so_the_call_stays_readable(self) -> None:
        messages = self._history()
        messages[1]["tool_calls"][0]["function"]["arguments"]["overwrite"] = True
        _digest_failed_call_args(messages)
        args = messages[1]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args["path"], "snap.py")
        self.assertIs(args["overwrite"], True)


class DigestShapeTests(unittest.TestCase):
    def test_arguments_stay_a_dict(self) -> None:
        """A JSON string would be replaced by {} on the Anthropic path."""
        out = _digest_call_args({"path": "a.py", "content": "x" * 900})
        self.assertIsInstance(out, dict)

    def test_nothing_to_elide_returns_the_same_object(self) -> None:
        args = {"path": "a.py", "overwrite": True}
        self.assertIs(_digest_call_args(args), args)

    def test_a_json_string_payload_is_parsed_not_mangled(self) -> None:
        out = _digest_call_args(json.dumps({"path": "a.py", "content": "x" * 900}))
        self.assertIsInstance(out, dict)
        self.assertEqual(out["path"], "a.py")

    def test_call_ids_survive_so_pairing_still_reconciles(self) -> None:
        messages = FailedCallDigestTests._history()
        _digest_failed_call_args(messages)
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "rejected")
        self.assertEqual(reconcile_tool_pairs(list(messages)), messages)

    def test_the_source_arguments_are_never_mutated_in_place(self) -> None:
        """The archive shares these nested dicts; a digest must not reach into them."""
        messages = FailedCallDigestTests._history()
        original_call = messages[1]["tool_calls"][0]
        original_args = original_call["function"]["arguments"]
        _digest_failed_call_args(messages)
        self.assertEqual(original_args["content"], "x" * 2_000)
        self.assertIsNot(messages[1]["tool_calls"][0], original_call)


class EnforceBudgetIntegrationTests(unittest.TestCase):
    def test_failed_arguments_are_dropped_regardless_of_the_budget(self) -> None:
        messages = FailedCallDigestTests._history()
        emitted: list[dict] = []
        with patch.object(history_module, "emit", emitted.append), \
             patch.object(history_module, "context_budget_for",
                          lambda model, mode: (1_000_000, 200_000, 500_000, 400_000)):
            history_module._enforce_context_budget(
                messages, "sys", None, {}, "m", "full", None, lambda t: len(t) // 4,
            )
        self.assertIn("elided", messages[1]["tool_calls"][0]["function"]["arguments"]["content"])
        self.assertTrue(any("failed tool call" in e.get("text", "") for e in emitted))


if __name__ == "__main__":
    unittest.main()

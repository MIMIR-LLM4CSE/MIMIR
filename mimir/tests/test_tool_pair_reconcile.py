"""Tests for the assistant↔tool pairing invariant (no network).

Three upstream mechanisms legitimately break the pairing — tool-result eviction,
intra-query compaction, and the dispatcher dropping duplicate calls — and every
backend receives the same history, so the repair belongs to ``history.py`` rather
than to one provider. These tests pin the repair itself, its wiring into the budget
pass, and the fact that each backend's own preparation preserves it.
"""

import json
import unittest

from mimir.client.query_engine.history import (
    EVICTED_TOOL_RESULT,
    _enforce_context_budget,
    _trim_tool_history,
    reconcile_tool_pairs,
)


def _assistant(*ids, content=""):
    """An assistant turn declaring one tool call per id (None = no id, Ollama-style)."""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            ({"id": i} if i is not None else {}) | {"function": {"name": "t", "arguments": {}}}
            for i in ids
        ],
    }


def _tool(call_id, content="ok"):
    msg = {"role": "tool", "content": content}
    if call_id is not None:
        msg["tool_call_id"] = call_id
    return msg


def _pairs_are_valid(messages) -> bool:
    """Every declared call answered, and no tool message without a preceding call."""
    i = 0
    while i < len(messages):
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared = len(m["tool_calls"])
            j = i + 1
            run = 0
            while j < len(messages) and messages[j].get("role") == "tool":
                run += 1
                j += 1
            if run != declared:
                return False
            i = j
        elif m.get("role") == "tool":
            return False  # orphan
        else:
            i += 1
    return True


class ReconcileTests(unittest.TestCase):
    def test_evicted_result_is_replaced_by_a_stub(self) -> None:
        out = reconcile_tool_pairs([
            _assistant("c1", "c2"),
            _tool("c2"),  # c1's result was evicted
        ])
        self.assertTrue(_pairs_are_valid(out))
        self.assertEqual(out[1]["tool_call_id"], "c1")
        self.assertEqual(out[1]["content"], EVICTED_TOOL_RESULT)
        self.assertEqual(out[2]["content"], "ok")

    def test_stub_payload_is_a_readable_error(self) -> None:
        payload = json.loads(EVICTED_TOOL_RESULT)
        self.assertEqual(payload["status"], "error")
        self.assertIn("evicted", payload["error"])

    def test_orphan_tool_message_is_dropped(self) -> None:
        out = reconcile_tool_pairs([
            {"role": "user", "content": "hi"},
            _tool("gone"),  # its assistant turn was compacted away
            {"role": "assistant", "content": "done"},
        ])
        self.assertTrue(_pairs_are_valid(out))
        self.assertEqual([m["role"] for m in out], ["user", "assistant"])

    def test_leading_orphan_from_front_trimming_is_dropped(self) -> None:
        # What a front-end's token-budget pop leaves behind.
        out = reconcile_tool_pairs([
            {"role": "system", "content": "S"},
            _tool("c1"),
            {"role": "user", "content": "next question"},
        ])
        self.assertEqual([m["role"] for m in out], ["system", "user"])

    def test_unidentified_calls_match_positionally(self) -> None:
        # Ollama emits tool_calls with no id; the dispatcher synthesises ids only on
        # the tool messages, so id matching cannot work and position must.
        out = reconcile_tool_pairs([
            _assistant(None, None),
            _tool("call_0", "first"),
            _tool("call_1", "second"),
        ])
        self.assertTrue(_pairs_are_valid(out))
        self.assertEqual([m["content"] for m in out[1:]], ["first", "second"])

    def test_unidentified_call_missing_a_result_gets_a_stub_without_id(self) -> None:
        out = reconcile_tool_pairs([_assistant(None, None), _tool("call_0")])
        self.assertTrue(_pairs_are_valid(out))
        self.assertNotIn("tool_call_id", out[2])
        self.assertEqual(out[2]["content"], EVICTED_TOOL_RESULT)

    def test_surplus_tool_messages_are_dropped(self) -> None:
        out = reconcile_tool_pairs([_assistant("c1"), _tool("c1"), _tool("c2"), _tool("c3")])
        self.assertTrue(_pairs_are_valid(out))
        self.assertEqual(len(out), 2)

    def test_duplicate_call_skipped_by_the_dispatcher_is_repaired(self) -> None:
        # The dispatcher skips an exact duplicate (name, args) within one step but the
        # assistant turn still declares both calls — a break with no trimming involved.
        out = reconcile_tool_pairs([_assistant("c1", "c2"), _tool("c1")])
        self.assertTrue(_pairs_are_valid(out))
        self.assertEqual(out[2]["content"], EVICTED_TOOL_RESULT)

    def test_intact_history_is_unchanged(self) -> None:
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "q"},
            _assistant("c1"),
            _tool("c1"),
            {"role": "assistant", "content": "answer"},
        ]
        self.assertEqual(reconcile_tool_pairs(list(messages)), messages)

    def test_reconciling_twice_is_stable(self) -> None:
        once = reconcile_tool_pairs([_assistant("c1", "c2"), _tool("c2")])
        self.assertEqual(reconcile_tool_pairs(list(once)), once)

    def test_assistant_without_tool_calls_passes_through(self) -> None:
        messages = [{"role": "assistant", "content": "hi", "tool_calls": []}]
        self.assertEqual(reconcile_tool_pairs(list(messages)), messages)

    def test_empty_history(self) -> None:
        self.assertEqual(reconcile_tool_pairs([]), [])


class TrimBreaksPairingTests(unittest.TestCase):
    """The break is real: eviction preserves the assistant turn by design."""

    def test_trim_strands_an_assistant_call(self) -> None:
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "q"},
            _assistant("c1"),
            _tool("c1", "x" * 500),
            _assistant("c2"),
            _tool("c2", "y" * 500),
        ]
        _trim_tool_history(messages, char_budget=600, execution_context={})
        self.assertFalse(_pairs_are_valid(messages))
        self.assertTrue(_pairs_are_valid(reconcile_tool_pairs(messages)))

    def test_budget_pass_leaves_history_paired(self) -> None:
        messages = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "q"},
            _assistant("c1"),
            _tool("c1", "x" * 4000),
            _assistant("c2"),
            _tool("c2", "y" * 4000),
        ]
        _enforce_context_budget(
            messages,
            system_content="S",
            step_tools=None,
            execution_context={},
            model="test-model",
            context_mode="compact",
            compact_fn=None,
            token_counter=lambda t: max(1, len(t) // 4),
        )
        self.assertTrue(_pairs_are_valid(messages))


class BackendPreparationTests(unittest.TestCase):
    """Each backend's own preparation must keep a reconciled history coherent."""

    def _broken(self):
        return [
            {"role": "user", "content": "q"},
            _assistant("c1", "c2"),
            _tool("c2", "res2"),  # c1's result evicted
        ]

    def test_vllm_preparation_pairs_every_call(self) -> None:
        from mimir.client.query_engine.backends.vllm_backend import (
            _prepare_messages_for_openai,
        )
        prepared = _prepare_messages_for_openai(self._broken())
        self.assertTrue(_pairs_are_valid(prepared))
        tool_ids = [m["tool_call_id"] for m in prepared if m["role"] == "tool"]
        self.assertEqual(tool_ids, ["c1", "c2"])

    def test_anthropic_preparation_pairs_every_tool_use(self) -> None:
        from mimir.client.query_engine.backends.anthropic_backend import AnthropicBackend
        _, conv = AnthropicBackend()._prepare(self._broken())
        used = [
            b["id"]
            for m in conv if m["role"] == "assistant"
            for b in m["content"] if b.get("type") == "tool_use"
        ]
        results = [
            b["tool_use_id"]
            for m in conv if m["role"] == "user" and isinstance(m["content"], list)
            for b in m["content"] if b.get("type") == "tool_result"
        ]
        self.assertEqual(sorted(used), ["c1", "c2"])
        self.assertEqual(sorted(results), ["c1", "c2"])

    def test_anthropic_drops_a_result_with_no_tool_use(self) -> None:
        from mimir.client.query_engine.backends.anthropic_backend import AnthropicBackend
        _, conv = AnthropicBackend()._prepare([
            {"role": "user", "content": "q"},
            _tool("gone", "orphan"),
            {"role": "assistant", "content": "done"},
        ])
        blocks = [b for m in conv if isinstance(m["content"], list) for b in m["content"]]
        self.assertEqual([b for b in blocks if b.get("type") == "tool_result"], [])


if __name__ == "__main__":
    unittest.main()

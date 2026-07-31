"""Tests for the Anthropic backend's request-shaping helpers (no network)."""

import unittest

import logging
import types

from mimir.client.query_engine.backends.anthropic_backend import (
    AnthropicBackend,
    _log_cache_usage,
    _mark_last_block_cacheable,
    _tools_to_anthropic,
)


class ToolConversionTests(unittest.TestCase):
    def test_ollama_function_to_claude_input_schema(self) -> None:
        tools = [{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }]
        out = _tools_to_anthropic(tools)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "read_file")
        self.assertEqual(out[0]["description"], "Read a file")
        self.assertIn("input_schema", out[0])
        self.assertEqual(out[0]["input_schema"]["properties"]["path"]["type"], "string")

    def test_malformed_tools_are_skipped(self) -> None:
        self.assertEqual(_tools_to_anthropic([{"type": "function"}, {}, {"function": {}}]), [])


class MessagePrepareTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = AnthropicBackend()

    def test_system_is_extracted_and_concatenated(self) -> None:
        system, conv = self.backend._prepare([
            {"role": "system", "content": "You are MIMIR."},
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
        ])
        self.assertEqual(system, "You are MIMIR.\n\nBe terse.")
        self.assertEqual(conv, [{"role": "user", "content": [{"type": "text", "text": "hi"}]}])

    def test_assistant_tool_calls_become_tool_use_blocks(self) -> None:
        _, conv = self.backend._prepare([
            {"role": "user", "content": "read x"},
            {
                "role": "assistant",
                "content": "sure",
                "tool_calls": [
                    {"id": "toolu_1", "function": {"name": "read_file", "arguments": {"path": "x"}}}
                ],
            },
        ])
        assistant = conv[1]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["content"][0], {"type": "text", "text": "sure"})
        tu = assistant["content"][1]
        self.assertEqual(tu["type"], "tool_use")
        self.assertEqual(tu["id"], "toolu_1")
        self.assertEqual(tu["name"], "read_file")
        self.assertEqual(tu["input"], {"path": "x"})

    def test_consecutive_tool_results_merge_into_one_user_turn(self) -> None:
        _, conv = self.backend._prepare([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "toolu_1", "function": {"name": "a", "arguments": {}}},
                {"id": "toolu_2", "function": {"name": "b", "arguments": {}}},
            ]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": "res1"},
            {"role": "tool", "tool_call_id": "toolu_2", "content": "res2"},
        ])
        # assistant turn, then a single user turn with two tool_result blocks
        self.assertEqual(conv[0]["role"], "assistant")
        self.assertEqual(conv[1]["role"], "user")
        blocks = conv[1]["content"]
        self.assertEqual([b["type"] for b in blocks], ["tool_result", "tool_result"])
        self.assertEqual(blocks[0]["tool_use_id"], "toolu_1")
        self.assertEqual(blocks[1]["content"], "res2")

    def test_non_string_tool_result_is_json_encoded(self) -> None:
        _, conv = self.backend._prepare([
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "toolu_1", "function": {"name": "a", "arguments": {}}}]},
            {"role": "tool", "tool_call_id": "toolu_1", "content": {"status": "ok"}},
        ])
        self.assertEqual(conv[1]["content"][0]["content"], '{"status": "ok"}')

    def test_cached_raw_blocks_replace_reconstructed_assistant_turn(self) -> None:
        # Simulate a prior response cached by tool-call id: its raw blocks (with a
        # signed thinking block) must be spliced back verbatim on replay.
        raw = ["<thinking-block>", "<tool_use-block>"]
        self.backend._raw_assistant_blocks[("toolu_9",)] = raw
        _, conv = self.backend._prepare([
            {"role": "assistant", "content": "flattened text", "tool_calls": [
                {"id": "toolu_9", "function": {"name": "a", "arguments": {}}}]},
        ])
        self.assertEqual(conv[0], {"role": "assistant", "content": raw})


class CacheBreakpointTests(unittest.TestCase):
    def test_last_dict_block_gets_cache_control(self) -> None:
        conv = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        _mark_last_block_cacheable(conv)
        self.assertEqual(
            conv[-1]["content"][-1]["cache_control"], {"type": "ephemeral"})

    def test_only_last_block_is_marked(self) -> None:
        conv = [{"role": "user", "content": [
            {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]
        _mark_last_block_cacheable(conv)
        self.assertNotIn("cache_control", conv[-1]["content"][0])
        self.assertIn("cache_control", conv[-1]["content"][1])

    def test_raw_sdk_block_is_not_mutated(self) -> None:
        # Replayed assistant turns hold non-dict SDK objects — must be left alone.
        conv = [{"role": "assistant", "content": ["<raw-sdk-block>"]}]
        _mark_last_block_cacheable(conv)  # no exception, no mutation
        self.assertEqual(conv[-1]["content"], ["<raw-sdk-block>"])

    def test_empty_conversation_is_safe(self) -> None:
        _mark_last_block_cacheable([])  # must not raise


class CacheUsageLogTests(unittest.TestCase):
    @staticmethod
    def _msg(**usage):
        return types.SimpleNamespace(usage=types.SimpleNamespace(**usage))

    def test_logs_hit_rate_from_usage(self) -> None:
        msg = self._msg(cache_read_input_tokens=90, cache_creation_input_tokens=0,
                        input_tokens=10, output_tokens=5)
        with self.assertLogs(
            "mimir.client.query_engine.backends.anthropic_backend", level="INFO"
        ) as cm:
            _log_cache_usage(msg)
        line = "\n".join(cm.output)
        self.assertIn("hit_rate=0.90", line)
        self.assertIn("read=90", line)

    def test_no_log_when_prompt_total_zero(self) -> None:
        # An empty usage block must not emit a divide-by-zero or a bogus line.
        logger = logging.getLogger(
            "mimir.client.query_engine.backends.anthropic_backend")
        with self.assertRaises(AssertionError):
            with self.assertLogs(logger, level="INFO"):
                _log_cache_usage(self._msg(
                    cache_read_input_tokens=0, cache_creation_input_tokens=0,
                    input_tokens=0, output_tokens=0))

    def test_missing_usage_is_safe(self) -> None:
        _log_cache_usage(types.SimpleNamespace())  # must not raise


if __name__ == "__main__":
    unittest.main()

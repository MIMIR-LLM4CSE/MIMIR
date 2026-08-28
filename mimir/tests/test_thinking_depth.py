"""Adaptive thinking: the depth ladder and the prompt directive that backs "auto".

The ladder (config/constants.py) is the single source of truth for the reasoning
depth; the agent derives `thinking` / `thinking_budget` from it, and only the "auto"
rung adds a calibration directive to the system prompt.
"""
import asyncio
import types
import unittest
from unittest.mock import patch

from mimir.tests._fake_backend import ScriptedBackend
from mimir.client.query_engine import agent_loop as agent_loop_module
from mimir.client.query_engine import finalize as finalize_module
from mimir.client.query_engine import history as history_module
from mimir.client.query_engine import streaming as streaming_module
from mimir.client.config import (
    DEFAULT_THINKING_DEPTH,
    THINKING_DEPTH_AUTO,
    THINKING_DEPTH_BUDGETS,
    THINKING_DEPTH_LABELS,
    clamp_thinking_depth,
    thinking_depth_from_label,
)
from mimir.client.agent_core import MimirAgent
from mimir.client.prompt import system_prompt as sp


class _Agent:
    """Minimal stand-in: the real setters over just the thinking state, so the
    ladder is exercised without standing up a whole agent (servers, backend…)."""

    set_thinking_depth = MimirAgent.set_thinking_depth
    set_thinking = MimirAgent.set_thinking
    set_thinking_budget = MimirAgent.set_thinking_budget

    def __init__(self):
        self.thinking_depth = DEFAULT_THINKING_DEPTH
        self.thinking = self.thinking_depth > 0
        self.thinking_budget = THINKING_DEPTH_BUDGETS[self.thinking_depth]


class LadderTest(unittest.TestCase):
    def test_labels_and_budgets_are_aligned(self):
        self.assertEqual(len(THINKING_DEPTH_LABELS), len(THINKING_DEPTH_BUDGETS))
        self.assertEqual(THINKING_DEPTH_LABELS[THINKING_DEPTH_AUTO], "auto")
        self.assertEqual(THINKING_DEPTH_LABELS[0], "off")

    def test_auto_is_unbudgeted(self):
        # "auto" must not push a thinking_budget: the depth is the model's to choose.
        self.assertEqual(THINKING_DEPTH_BUDGETS[THINKING_DEPTH_AUTO], -1)

    def test_clamp(self):
        self.assertEqual(clamp_thinking_depth(-5), 0)
        self.assertEqual(clamp_thinking_depth(99), len(THINKING_DEPTH_LABELS) - 1)
        self.assertEqual(clamp_thinking_depth(2), 2)

    def test_label_resolution(self):
        for i, label in enumerate(THINKING_DEPTH_LABELS):
            self.assertEqual(thinking_depth_from_label(label), i)
            self.assertEqual(thinking_depth_from_label(label.upper()), i)
        self.assertEqual(thinking_depth_from_label("on"), THINKING_DEPTH_AUTO)
        self.assertEqual(thinking_depth_from_label("off"), 0)
        self.assertIsNone(thinking_depth_from_label("deeper"))


class AgentStateTest(unittest.TestCase):
    def test_default_is_auto(self):
        a = _Agent()
        self.assertEqual(a.thinking_depth, THINKING_DEPTH_AUTO)
        self.assertTrue(a.thinking)
        self.assertEqual(a.thinking_budget, -1)

    def test_every_rung_yields_a_coherent_triplet(self):
        a = _Agent()
        for level in range(len(THINKING_DEPTH_LABELS)):
            a.set_thinking_depth(level)
            self.assertEqual(a.thinking_depth, level)
            self.assertEqual(a.thinking, level > 0)
            self.assertEqual(a.thinking_budget, THINKING_DEPTH_BUDGETS[level])

    def test_out_of_range_is_clamped(self):
        a = _Agent()
        a.set_thinking_depth(-3)
        self.assertEqual(a.thinking_depth, 0)
        a.set_thinking_depth(42)
        self.assertEqual(a.thinking_depth, len(THINKING_DEPTH_LABELS) - 1)

    def test_legacy_switch_maps_on_to_auto(self):
        a = _Agent()
        a.set_thinking(False)
        self.assertEqual(a.thinking_depth, 0)
        self.assertFalse(a.thinking)
        a.set_thinking(True)
        self.assertEqual(a.thinking_depth, THINKING_DEPTH_AUTO)
        self.assertTrue(a.thinking)

    def test_budget_override_does_not_move_the_rung(self):
        a = _Agent()
        a.set_thinking_depth(THINKING_DEPTH_AUTO)
        a.set_thinking_budget(2048)
        self.assertEqual(a.thinking_budget, 2048)
        self.assertEqual(a.thinking_depth, THINKING_DEPTH_AUTO)


def _prompt(depth: int) -> str:
    return sp.build_system_content(
        active_mode="agent",
        tool_owner={},
        sensitive_tools=set(),
        thinking_depth=depth,
    )


class PromptDirectiveTest(unittest.TestCase):
    def test_auto_injects_the_directive(self):
        self.assertIn(sp._THINKING_DIRECTIVE_AUTO, _prompt(THINKING_DEPTH_AUTO))

    def test_other_rungs_share_one_identical_prompt(self):
        others = [d for d in range(len(THINKING_DEPTH_LABELS)) if d != THINKING_DEPTH_AUTO]
        rendered = {d: _prompt(d) for d in others}
        for d, text in rendered.items():
            self.assertNotIn(sp._THINKING_DIRECTIVE_AUTO, text, f"depth {d}")
        # Byte-identical across every non-auto rung: no prefix-cache churn there.
        self.assertEqual(len(set(rendered.values())), 1)

    def test_calibration_rule_is_always_present(self):
        # The proportionality rule lives in the base section, so it applies at
        # every rung — including "off", where it governs the visible reasoning.
        for depth in range(len(THINKING_DEPTH_LABELS)):
            self.assertIn("Match the depth to the stakes", _prompt(depth))


# --------------------------------------------------------------------------
# Live (mid-query) rung changes
# --------------------------------------------------------------------------

class _LoopAgent:
    """Stub with the real depth setters, enough surface to drive run_agent_query."""

    set_thinking_depth = MimirAgent.set_thinking_depth
    mode = "agent"
    model = "dummy"
    tools: list = []
    tool_owner: dict = {}
    tool_caps: dict = {}
    thinking_budget = -1
    allow_continue_prompt = False
    _cancel_flag = None
    approvals = types.SimpleNamespace(flush_pending_review=lambda: None)

    def __init__(self):
        self.thinking_depth = DEFAULT_THINKING_DEPTH
        self.thinking = True
        self.thinking_budget = -1

    # The only depth-dependent part of the prompt, stubbed so the rewrite is visible.
    async def _build_system_content(self, **kwargs):
        return f"SYS[auto={self.thinking_depth == THINKING_DEPTH_AUTO}]"

    @staticmethod
    def _new_execution_context():
        return MimirAgent._new_execution_context()

    @staticmethod
    def _get_todo_file():
        return ""

    def _apply_carry_context(self, execution_context):
        pass

    def _update_carry_context(self, execution_context):
        pass

    @staticmethod
    def _normalize_mode(mode):
        return "agent"

    @staticmethod
    def _normalize_arguments(args):
        return args

    @staticmethod
    def _truncate_text(text, limit=600):
        return text[:limit]

    async def _run_tool(self, tool, args, execution_context=None, run_auto_validation=True, call_id=""):
        return "{}"


def _tool_call(name):
    return {"id": "1", "function": {"name": name, "arguments": "{}"}}


class LiveRungChangeTests(unittest.TestCase):
    """A rung moved mid-query must land on the very next model call."""

    def _run(self, agent, script, on_step):
        backend = ScriptedBackend(script)

        async def _flip(*a, **k):
            on_step()
            return None

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(m, "_dispatch_tool_calls", _flip), \
             patch.object(m, "_post_dispatch_inject", _noop_async), \
             patch.object(history_module, "_trim_tool_history", lambda *a, **k: None), \
             patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: []), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            asyncio.run(m.run_agent_query(agent=agent, query="do a thing", max_steps=5, thinking=True))
        return backend

    def test_turning_thinking_off_mid_query_reaches_the_next_call(self):
        agent = _LoopAgent()
        script = [
            {"content": "working", "tool_calls": [_tool_call("noop")]},
            {"content": "done"},
        ]
        backend = self._run(agent, script, lambda: agent.set_thinking_depth(0))

        self.assertEqual(len(backend.calls), 2)
        self.assertTrue(backend.calls[0]["thinking"])
        self.assertFalse(backend.calls[1]["thinking"])  # picked up mid-query

    def test_leaving_auto_rewrites_the_system_message(self):
        agent = _LoopAgent()
        script = [
            {"content": "working", "tool_calls": [_tool_call("noop")]},
            {"content": "done"},
        ]
        backend = self._run(agent, script, lambda: agent.set_thinking_depth(3))

        self.assertEqual(backend.calls[0]["messages"][0]["content"], "SYS[auto=True]")
        self.assertEqual(backend.calls[1]["messages"][0]["content"], "SYS[auto=False]")

    def test_moving_to_a_budgeted_rung_starts_budgeting_mid_query(self):
        agent = _LoopAgent()
        script = [
            {"content": "working", "tool_calls": [_tool_call("noop")]},
            {"content": "done"},
        ]
        # "medium" == 4096, unscaled in the default (discover) workflow phase.
        backend = self._run(agent, script, lambda: agent.set_thinking_depth(3))

        self.assertNotIn("thinking_budget", backend.calls[0]["options"])
        self.assertEqual(backend.calls[1]["options"]["thinking_budget"], 4096)

    def test_steady_state_leaves_the_system_message_untouched(self):
        agent = _LoopAgent()
        script = [
            {"content": "working", "tool_calls": [_tool_call("noop")]},
            {"content": "done"},
        ]
        backend = self._run(agent, script, lambda: None)

        # No rung change → byte-stable prefix, as before.
        self.assertEqual(
            backend.calls[0]["messages"][0]["content"],
            backend.calls[1]["messages"][0]["content"],
        )


class StubFallbackTests(unittest.TestCase):
    """Callers driving a bare stub keep the flag they passed in."""

    def test_live_thinking_falls_back_without_the_attribute(self):
        bare = types.SimpleNamespace(model="m")
        self.assertTrue(agent_loop_module._live_thinking(bare, True))
        self.assertFalse(agent_loop_module._live_thinking(bare, False))

    def test_directive_sync_is_a_noop_without_a_depth(self):
        bare = types.SimpleNamespace(model="m")
        messages = [{"role": "system", "content": "S"}]
        auto, sys_content = asyncio.run(
            agent_loop_module._sync_thinking_directive(bare, messages, "agent", False, "S")
        )
        self.assertFalse(auto)
        self.assertEqual(sys_content, "S")
        self.assertEqual(messages[0]["content"], "S")


if __name__ == "__main__":
    unittest.main()

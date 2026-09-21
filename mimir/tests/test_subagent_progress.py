"""What a delegating tool reports while it runs, and how it reaches the UI.

A delegated run is one tool call that lasts minutes. Without a channel out of it the
frontend has a spinner and nothing else, so the child's tool activity travels back as
MCP progress notifications and is re-emitted as rows under the parent's row. Tested
here: the gate (no row, no frontend → no listener), the envelope contract, and the
isolation between concurrent delegations.

Pure-Python + stubs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import asyncio
import json
import types
import unittest
from unittest import mock

from mimir.client.event_sink import event_sink
from mimir.client.tool_execution import executor as ex


def _drive(cb, payload) -> None:
    """Feed one progress notification to a callback the way the MCP session does."""
    message = payload if isinstance(payload, str) else json.dumps(payload)
    asyncio.run(cb(1.0, None, message))


class ListenerGateTests(unittest.TestCase):
    def test_no_listener_without_a_frontend(self):
        """The CLI prints every event as a raw JSON line; multiplying that by every
        child tool call would be noise, not a view."""
        self.assertIsNone(ex._make_subagent_progress_cb("c1"))

    def test_no_listener_without_a_row_to_hang_the_activity_on(self):
        with event_sink(lambda ev: None):
            self.assertIsNone(ex._make_subagent_progress_cb(""))

    def test_a_listener_when_both_are_there(self):
        with event_sink(lambda ev: None):
            self.assertIsNotNone(ex._make_subagent_progress_cb("c1"))


class ContextTests(unittest.TestCase):
    def test_events_reach_the_frontend_from_the_session_s_own_task(self):
        """The callback fires on the MCP session's receive loop — a task started at
        connect time, whose context snapshot predates the run's event sink. Reading
        the sink there would find nothing and print the event instead of showing it,
        so the sink is captured where it is bound."""
        seen: list[dict] = []
        with event_sink(seen.append):
            cb = ex._make_subagent_progress_cb("call_7")
        # Outside the binding, as the receive loop effectively is.
        _drive(cb, {"v": 1, "t": "tc", "i": "c1", "n": "grep"})
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["parent_id"], "call_7")

    def test_a_frontend_that_throws_is_not_the_tool_call_s_problem(self):
        def _boom(_ev):
            raise RuntimeError("ui gone")

        with event_sink(_boom):
            cb = ex._make_subagent_progress_cb("call_7")
            _drive(cb, {"v": 1, "t": "tc", "i": "c1", "n": "grep"})  # must not raise


class EnvelopeTests(unittest.TestCase):
    def _emitted(self, payload) -> list[dict]:
        seen: list[dict] = []
        with event_sink(seen.append):
            _drive(ex._make_subagent_progress_cb("call_7"), payload)
        return seen

    def test_a_child_tool_call_becomes_a_row_under_its_parent(self):
        events = self._emitted(
            {"v": 1, "t": "tc", "i": "c1", "n": "read_file_lines",
             "l": "Reading file: agent_core.py", "d": "1-200"})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "subagent_event")
        self.assertEqual(events[0]["kind"], "tool_call")
        self.assertEqual(events[0]["parent_id"], "call_7")
        self.assertEqual(events[0]["name"], "read_file_lines")
        self.assertEqual(events[0]["label"], "Reading file: agent_core.py")

    def test_a_child_result_carries_the_same_id_as_its_call(self):
        call = self._emitted({"v": 1, "t": "tc", "i": "c1", "n": "grep"})[0]
        result = self._emitted(
            {"v": 1, "t": "tr", "i": "c1", "ok": True, "s": "3 matches", "ms": 41})[0]
        self.assertEqual(result["id"], call["id"])
        self.assertTrue(result["ok"])
        self.assertEqual(result["duration_ms"], 41)

    def test_child_ids_are_namespaced_under_the_parent(self):
        """Child ids come from the child's own model, so two concurrent explorers
        both produce "c1" — un-namespaced, one would patch the other's row."""
        first = self._emitted({"v": 1, "t": "tc", "i": "c1", "n": "grep"})[0]
        seen: list[dict] = []
        with event_sink(seen.append):
            _drive(ex._make_subagent_progress_cb("call_9"),
                   {"v": 1, "t": "tc", "i": "c1", "n": "grep"})
        self.assertNotEqual(first["id"], seen[0]["id"])
        self.assertTrue(first["id"].startswith("call_7"))
        self.assertTrue(seen[0]["id"].startswith("call_9"))

    def test_a_shed_event_count_comes_back_as_a_trailer(self):
        events = self._emitted({"v": 1, "t": "end", "dropped": 12})
        self.assertEqual(events[0]["kind"], "end")
        self.assertEqual(events[0]["dropped"], 12)

    def test_progress_that_is_not_ours_is_ignored(self):
        """Any server may report progress; only our envelope means sub-agent activity."""
        for payload in (
            "rendering 40%",                       # a plain human message
            "",                                    # no message at all
            {"percent": 40},                       # someone else's JSON
            {"v": 2, "t": "tc", "i": "c1"},        # a future schema
            {"v": 1, "t": "unknown", "i": "c1"},   # a kind we don't render
        ):
            with self.subTest(payload=payload):
                self.assertEqual(self._emitted(payload), [])


class CallSiteTests(unittest.IsolatedAsyncioTestCase):
    """The listener is attached to the actual tool call, unconditionally."""

    async def _call(self, call_id: str, tool_caps: dict | None = None):
        captured: dict = {}

        class _Session:
            async def call_tool(self, name, args, **kwargs):
                captured.update(kwargs)
                return types.SimpleNamespace(content=[])

        agent = types.SimpleNamespace(
            model="served/model",
            sessions={"srv": _Session()},
            tool_owner={"delegate": "srv", "read_thing": "srv"},
            mode="agent",
            tool_caps=tool_caps or {},
            disabled_servers=set(),
            _tool_cache={},
            approvals=types.SimpleNamespace(batch_mode=False, approval_mode="auto"),
            _normalize_tool_content=lambda r: "{}",
            _parse_tool_payload=lambda t: {},
            _normalize_workspace_path=lambda p: p,
            tools=[{"type": "function", "function": {"name": n}}
                   for n in ("delegate", "read_thing")],
        )
        agent.advertised_tools = lambda: agent.tools
        evaluation = types.SimpleNamespace(
            violation=None, tool_name="delegate", arguments={}, execution_context={},
        )
        with mock.patch.object(ex, "evaluate_tool_preconditions", return_value=evaluation), \
             mock.patch.object(ex, "record_tool_observation"), \
             mock.patch.object(ex.bash_effect, "capture", return_value=None), \
             mock.patch.object(ex.bash_effect, "report", return_value=""), \
             mock.patch.object(ex.bash_effect, "created_paths", return_value=[]):
            await ex.execute_tool_call(
                agent=agent, tool_name="delegate", arguments={},
                execution_context={}, run_auto_validation=False, call_id=call_id,
            )
        return captured

    async def test_the_call_carries_a_listener_when_a_frontend_is_bound(self):
        with event_sink(lambda ev: None):
            captured = await self._call("call_7")
        self.assertIsNotNone(captured.get("progress_callback"))

    async def test_the_call_still_goes_through_without_one(self):
        """A server that never reports progress must pay nothing for this."""
        captured = await self._call("call_7")
        self.assertIn("progress_callback", captured)
        self.assertIsNone(captured["progress_callback"])

    async def test_the_call_carries_the_approval_mode_and_the_model(self):
        """A sub-agent must run in the mode the user chose and default to the model in
        use, both read at call time."""
        from mimir.client.config.models import CALLER_MODEL_META
        from mimir.client.guardrails.policy.approval import APPROVAL_MODE_META
        captured = await self._call("call_7")
        self.assertEqual(captured["meta"],
                         {APPROVAL_MODE_META: "auto", CALLER_MODEL_META: "served/model"})

    async def test_only_a_delegating_call_carries_what_it_may_grant(self):
        """The table is for the one tool that hands tools on. On every other call it
        would be payload nobody reads."""
        from mimir.client.config.models import CALLER_MODE_META, GRANTABLE_TOOLS_META
        from mimir.client.context.capabilities import DELEGATE, ToolCaps
        captured = await self._call("call_7")
        self.assertNotIn(GRANTABLE_TOOLS_META, captured["meta"])

        captured = await self._call(
            "call_8",
            tool_caps={"delegate": ToolCaps(name="delegate",
                                            capabilities=frozenset({DELEGATE}))})
        # Delegation excludes itself: a child that can delegate spawns a tree nobody
        # budgeted.
        self.assertEqual(captured["meta"][GRANTABLE_TOOLS_META], {"read_thing": "srv"})
        self.assertEqual(captured["meta"][CALLER_MODE_META], "agent")


if __name__ == "__main__":
    unittest.main()

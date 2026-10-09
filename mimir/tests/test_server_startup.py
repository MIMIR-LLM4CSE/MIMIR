"""Hermetic tests for connecting a whole server registry at once.

Start-up used to be the sum of every server's: twenty child interpreters importing
the MCP SDK, one after the next, before the first query could be answered. The
processes are now started together and the handshakes awaited at once, which puts
the cost at the slowest server rather than the total. What these tests pin is what
that overlap must not change — the order the model sees the tools in, and a
caller's tolerance of a server that fails to come up.
"""

import asyncio
import time
import types
import unittest
from unittest.mock import patch

from mimir.client.integration import server_manager


def _run(coro):
    return asyncio.run(coro)


class _FakeTool:
    def __init__(self, name):
        self.name = name
        self.description = ""
        self.inputSchema = {"type": "object", "properties": {}}
        self.meta = None
        self.annotations = None


class _FakeSession:
    """A session whose handshake takes *delay* seconds, or fails."""

    def __init__(self, tool_name, delay=0.0, fail=False):
        self.tool_name = tool_name
        self.delay = delay
        self.fail = fail

    async def initialize(self):
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.tool_name}: handshake failed")

    async def list_tools(self):
        return types.SimpleNamespace(tools=[_FakeTool(self.tool_name)])

    async def list_resources(self):
        return types.SimpleNamespace(resources=[])


class _FakeAgent:
    def __init__(self):
        self.sessions = {}
        self.tools = []
        self.tool_owner = {}
        self.tool_caps = {}
        self.resources = {}


def _connect(agent, sessions, on_error=None):
    """Drive ``connect_servers`` over *sessions* — ``{server: _FakeSession}``."""
    async def fake_spawn(*, agent, name, script):
        session = sessions[name]
        if isinstance(session, BaseException):
            raise session
        return session

    with patch.object(server_manager, "_spawn_session", fake_spawn):
        return _run(server_manager.connect_servers(
            agent=agent,
            registry={name: f"{name}.py" for name in sessions},
            on_error=on_error,
        ))


class RegistryOrderTests(unittest.TestCase):
    """The tool schema is sent on every request, so its order is a cached prefix."""

    def test_tools_are_registered_in_registry_order_not_handshake_order(self):
        # Reverse-ordered delays: the last server in the registry answers first.
        sessions = {
            "alpha": _FakeSession("a_tool", delay=0.03),
            "beta": _FakeSession("b_tool", delay=0.02),
            "gamma": _FakeSession("c_tool", delay=0.01),
        }
        agent = _FakeAgent()
        _connect(agent, sessions)
        self.assertEqual([t["function"]["name"] for t in agent.tools],
                         ["a_tool", "b_tool", "c_tool"])
        self.assertEqual(list(agent.sessions), ["alpha", "beta", "gamma"])

    def test_the_handshakes_overlap(self):
        """The point of the exercise: the registry costs its slowest server."""
        sessions = {f"s{i}": _FakeSession(f"t{i}", delay=0.1) for i in range(8)}
        agent = _FakeAgent()
        started = time.monotonic()
        _connect(agent, sessions)
        elapsed = time.monotonic() - started
        self.assertEqual(len(agent.tools), 8)
        # Serialized this would be 0.8s. Generous, so a loaded machine cannot fail it
        # while still failing outright if the handshakes ever go back to sequential.
        self.assertLess(elapsed, 0.4)


class ToleratedFailureTests(unittest.TestCase):
    """A sub-agent connects a subset and must survive a server it does not need."""

    def test_a_failed_handshake_is_reported_and_the_others_still_connect(self):
        sessions = {
            "good": _FakeSession("good_tool"),
            "broken": _FakeSession("broken_tool", fail=True),
            "also_good": _FakeSession("also_good_tool"),
        }
        agent = _FakeAgent()
        seen = []
        _connect(agent, sessions, on_error=lambda n, e: seen.append((n, str(e))))
        self.assertEqual([t["function"]["name"] for t in agent.tools],
                         ["good_tool", "also_good_tool"])
        self.assertEqual([n for n, _ in seen], ["broken"])
        self.assertIn("handshake failed", seen[0][1])

    def test_a_failed_spawn_is_reported_and_the_others_still_connect(self):
        sessions = {
            "good": _FakeSession("good_tool"),
            "missing": ValueError("Server script must be a .py or .js file"),
        }
        agent = _FakeAgent()
        seen = []
        _connect(agent, sessions, on_error=lambda n, e: seen.append((n, str(e))))
        self.assertEqual([t["function"]["name"] for t in agent.tools], ["good_tool"])
        self.assertEqual([n for n, _ in seen], ["missing"])

    def test_without_a_handler_the_failure_reaches_the_caller(self):
        """The startup path: a half-built agent is closed, not run."""
        agent = _FakeAgent()
        with self.assertRaises(RuntimeError):
            _connect(agent, {"broken": _FakeSession("broken_tool", fail=True)})

        agent = _FakeAgent()
        with self.assertRaises(ValueError):
            _connect(agent, {"missing": ValueError("bad script")})


if __name__ == "__main__":
    unittest.main()

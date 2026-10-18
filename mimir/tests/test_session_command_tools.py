"""A slash command runs directly (``call_tool_directly``), outside the policy pipeline
meant for the model's calls; both surfaces share one command set.
"""

from __future__ import annotations

import asyncio
import os
import types
import unittest

from mimir.client.ui.ws.worker.worker import AgentWorker
from mimir.client.core.turn_state import TurnState
from mimir.tests.support.agents import fake_agent

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeSession:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool: str, args: dict):
        self.calls.append((tool, dict(args)))
        # As an MCP session answers: one text block carrying the payload.
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=self.payload)])


def _worker_stub(session: _FakeSession, loop) -> types.SimpleNamespace:
    """The minimum of ``AgentWorker`` that ``call_session_tool`` touches."""

    def _boom(*_a, **_k):
        raise AssertionError(
            "the guardrail pipeline was entered for a user-typed command")

    agent = fake_agent(
        tool_owner={"memory_list_all": "memory"},
        sessions={"memory": session},
        run_tool=_boom,turn=TurnState())
    return types.SimpleNamespace(_agent=agent, _loop=loop)


class CallSessionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread_done = None

    def tearDown(self) -> None:
        self.loop.close()

    def _run(self, worker, tool: str, args: dict) -> dict:
        """Drive the worker's loop until the scheduled coroutine resolves."""
        fut = AgentWorker.call_session_tool(worker, tool, args)
        # run_coroutine_threadsafe needs the loop to actually turn; this test owns it.
        while not fut.done():
            self.loop.call_soon(self.loop.stop)
            self.loop.run_forever()
        return fut.result()

    def test_it_calls_the_owning_server_directly(self) -> None:
        session = _FakeSession('{"status": "ok", "memory": [], "count": 0}')
        worker = _worker_stub(session, self.loop)

        payload = self._run(worker, "memory_list_all", {})

        # run_tool would have raised; the call went straight to the MCP session.
        self.assertEqual(session.calls, [("memory_list_all", {})])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 0)

    def test_an_unconnected_server_is_named_rather_than_raised(self) -> None:
        session = _FakeSession("{}")
        worker = _worker_stub(session, self.loop)

        payload = self._run(worker, "proxy_get", {"op": "proxies"})

        self.assertEqual(session.calls, [])
        self.assertEqual(payload["status"], "error")
        self.assertIn("proxy_get", payload["error"])

    def test_a_failing_call_becomes_an_error_payload(self) -> None:
        session = _FakeSession("not json")
        worker = _worker_stub(session, self.loop)

        payload = self._run(worker, "memory_list_all", {})

        self.assertEqual(payload["status"], "error")

    def test_no_agent_means_no_crash(self) -> None:
        worker = types.SimpleNamespace(_agent=None, _loop=None)
        fut = AgentWorker.call_session_tool(worker, "memory_list_all", {})
        self.assertEqual(fut.result()["status"], "error")


class ProxyListTests(unittest.TestCase):
    """`/proxy clean` needs a name; listing them must not require asking the model."""

    def test_both_surfaces_offer_it(self) -> None:
        from mimir.client.ui.cli.chat_commands import CLI_COMMANDS
        from mimir.client.ui.ws.connection.commands import WS_COMMANDS
        for registry in (CLI_COMMANDS, WS_COMMANDS):
            self.assertIsNotNone(registry.find("/proxy list"))

    def test_it_lists_through_the_registry_tool(self) -> None:
        from mimir.tests.support.commands import cli
        session = _FakeSession('{"status": "ok", "proxies": [{"name": "wave2d"}]}')
        agent = fake_agent(tool_owner={"proxy_get": "proxy"},
                                      sessions={"proxy": session}, turn=TurnState())
        self.assertIn("wave2d", cli("/proxy list", agent))
        self.assertEqual(session.calls, [("proxy_get", {"op": "proxies"})])

    def test_the_dropdown_advertises_it(self) -> None:
        with open(os.path.join(_ROOT, "vscode-extension", "webview", "src", "components",
                               "slashUtils.ts"), encoding="utf-8") as fh:
            src = fh.read()
        block = src.split("SESSION_COMMANDS", 1)[1].split("];", 1)[0]
        proxy = next(line for line in block.splitlines() if 'name: "proxy"' in line)
        self.assertIn("list", proxy)

    def test_the_usage_line_names_both_subcommands(self) -> None:
        # A usage line that omits one is how a working command stays undiscovered.
        from mimir.tests.support.commands import cli
        usage = cli("/proxy", object())
        self.assertIn("/proxy list", usage)
        self.assertIn("/proxy clean <name>", usage)

if __name__ == "__main__":
    unittest.main()

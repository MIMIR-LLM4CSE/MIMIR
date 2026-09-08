"""A slash command must not be judged as if the model had asked for it.

`call_session_tool` is what `/memory list`, `/memory clear` and `/proxy clean` run
on. It used to go through `agent._run_tool` — the full policy pipeline: approvals,
plan-shape gates, write policy. Those exist to weigh what the MODEL proposed. Applied
to a command the user typed themselves, the plan gate answered with a refusal string
in plan or ask mode, so the command did nothing, and each one scattered tool cards
through the transcript on its way there.

The CLI surface always bypassed the pipeline (`chat_commands._call_platform_tool`);
these tests pin the WS surface to the same route, and the two surfaces to the same
command set.

Run:
    python -m unittest mimir.tests.test_session_command_tools -v
"""

from __future__ import annotations

import asyncio
import os
import re
import types
import unittest

from mimir.client.ui.ws.ws_worker import _AgentWorker

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _FakeSession:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool: str, args: dict):
        self.calls.append((tool, dict(args)))
        return self.payload


def _worker_stub(session: _FakeSession, loop) -> types.SimpleNamespace:
    """The minimum of ``_AgentWorker`` that ``call_session_tool`` touches."""

    def _boom(*_a, **_k):
        raise AssertionError(
            "the guardrail pipeline was entered for a user-typed command")

    agent = types.SimpleNamespace(
        tool_owner={"memory_list_all": "memory"},
        sessions={"memory": session},
        _run_tool=_boom,
        _normalize_tool_content=lambda raw: raw,
    )
    return types.SimpleNamespace(_agent=agent, _loop=loop)


class CallSessionToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread_done = None

    def tearDown(self) -> None:
        self.loop.close()

    def _run(self, worker, tool: str, args: dict) -> dict:
        """Drive the worker's loop until the scheduled coroutine resolves."""
        fut = _AgentWorker.call_session_tool(worker, tool, args)
        # run_coroutine_threadsafe needs the loop to actually turn; this test owns it.
        while not fut.done():
            self.loop.call_soon(self.loop.stop)
            self.loop.run_forever()
        return fut.result()

    def test_it_calls_the_owning_server_directly(self) -> None:
        session = _FakeSession('{"status": "ok", "memory": [], "count": 0}')
        worker = _worker_stub(session, self.loop)

        payload = self._run(worker, "memory_list_all", {})

        # _run_tool would have raised; the call went straight to the MCP session.
        self.assertEqual(session.calls, [("memory_list_all", {})])
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["count"], 0)

    def test_an_unconnected_server_is_named_rather_than_raised(self) -> None:
        session = _FakeSession("{}")
        worker = _worker_stub(session, self.loop)

        fut = _AgentWorker.call_session_tool(worker, "proxy_get", {"op": "proxies"})

        self.assertEqual(session.calls, [])
        self.assertEqual(fut.result()["status"], "error")
        self.assertIn("proxy_get", fut.result()["error"])

    def test_a_failing_call_becomes_an_error_payload(self) -> None:
        session = _FakeSession("not json")
        worker = _worker_stub(session, self.loop)

        payload = self._run(worker, "memory_list_all", {})

        self.assertEqual(payload["status"], "error")

    def test_no_agent_means_no_crash(self) -> None:
        worker = types.SimpleNamespace(_agent=None, _loop=None)
        fut = _AgentWorker.call_session_tool(worker, "memory_list_all", {})
        self.assertEqual(fut.result()["status"], "error")


class ProxyListTests(unittest.TestCase):
    """`/proxy clean` needs a name; listing them must not require asking the model."""

    def _read(self, *parts: str) -> str:
        with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
            return fh.read()

    def test_the_ws_surface_handles_it(self) -> None:
        src = self._read("client", "ui", "ws", "ws_session.py")
        self.assertIn('parts[1] == "list"', src)
        self.assertIn('"proxy_get", {"op": "proxies"}', src)

    def test_the_cli_surface_handles_it(self) -> None:
        src = self._read("client", "ui", "cli", "chat_commands.py")
        self.assertIn("_proxy_list_command", src)
        self.assertIn('"proxy_get", {"op": "proxies"}', src)

    def test_the_dropdown_advertises_it(self) -> None:
        src = self._read("vscode-extension", "webview", "src", "components",
                         "slashUtils.ts")
        block = src.split("SESSION_COMMANDS", 1)[1].split("];", 1)[0]
        proxy = next(line for line in block.splitlines() if 'name: "proxy"' in line)
        self.assertIn("list", proxy)

    def test_the_usage_line_names_both_subcommands(self) -> None:
        # A usage line that omits one is how a working command stays undiscovered.
        for path in (("client", "ui", "ws", "ws_session.py"),
                     ("client", "ui", "cli", "chat_commands.py")):
            src = self._read(*path)
            usage = re.search(r"Usage: /proxy[^\"]*", src)
            self.assertIsNotNone(usage, path)
            self.assertIn("list", usage.group(0))


if __name__ == "__main__":
    unittest.main()

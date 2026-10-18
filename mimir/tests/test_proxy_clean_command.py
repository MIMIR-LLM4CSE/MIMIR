"""The ``/proxy clean`` slash command's surface; the server op is covered by the proxy op
tests.
"""

import os
import types
import unittest
from mimir.client.core.turn_state import TurnState

_PROXY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "servers", "proxy")


class ProxyCleanCommandTests(unittest.TestCase):
    """``/proxy clean <name>``: housekeeping the user can do without asking the model."""

    def _run(self, query: str, payload: dict):
        from unittest import mock
        from mimir.client.ui.cli import chat_commands
        from mimir.tests.support.commands import cli

        async def _fake(agent, tool, arguments):
            self.assertEqual(tool, "proxy_manage")
            self.assertEqual(arguments.get("op"), "clean")
            self.assertTrue(arguments.get("confirm"))
            return payload

        with mock.patch.object(chat_commands, "call_tool_directly", _fake):
            return cli(query, object())

    def test_it_reports_what_was_removed_and_what_survived(self) -> None:
        msg = self._run("/proxy clean wave2d", {
            "status": "ok",
            "removed": ["runs", "optimisation state"],
            "kept": ["sealed references wave2d_ref — shared with suites"],
        })
        self.assertIn("wave2d", msg)
        self.assertIn("runs", msg)
        # The surviving half is the answer to "I deleted everything and it still
        # remembers" — printing only what was removed would reproduce the confusion.
        self.assertIn("kept", msg)
        self.assertIn("sealed references", msg)

    def test_a_disconnected_proxy_server_says_so(self) -> None:
        from mimir.tests.support.commands import cli
        msg = cli("/proxy clean wave2d", types.SimpleNamespace(tool_owner={}, sessions={}, turn=TurnState()))
        self.assertIn("not connected", msg)

    def test_usage_without_a_name(self) -> None:
        from mimir.tests.support.commands import cli
        msg = cli("/proxy", object())
        self.assertIn("/proxy clean <name>", msg)

    def test_the_command_is_listed_in_help(self) -> None:
        from mimir.tests.support.commands import cli
        self.assertIn("/proxy clean", cli("/help"))


if __name__ == "__main__":
    unittest.main()

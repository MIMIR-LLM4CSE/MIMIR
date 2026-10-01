"""The ``/proxy clean`` slash command (client side).

Housekeeping the user can ask for without going through the model — and the reason
it exists: a store that outlives its project brings a finished checklist back with
it. The server op it drives is covered by the proxy op tests; what is checked here
is the command surface.

Run:
    python -m unittest mimir.tests.test_proxy_clean_command -v
"""

import os
import sys
import unittest

_PROXY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "servers", "proxy")
sys.path.insert(0, os.path.abspath(_PROXY))
sys.path.insert(0, os.path.abspath(os.path.join(_PROXY, "..", "_shared")))


class ProxyCleanCommandTests(unittest.TestCase):
    """`/proxy clean <name>` — housekeeping the user can do without asking the model.

    `clean` is reachable as a tool op, but a person who wants to start an optimisation
    over should not have to ask the model to tidy up first. The alternative is `rm -rf` on
    a store whose path nobody has a reason to know — which is how a deleted project comes
    back with a finished checklist and an optimisation still marked "in progress".
    """

    def _run(self, query: str, payload: dict | None):
        import asyncio
        from unittest import mock
        from mimir.client.ui.cli import chat_commands

        async def _fake(agent, tool, arguments):
            self.assertEqual(tool, "proxy_manage")
            self.assertEqual(arguments.get("op"), "clean")
            self.assertTrue(arguments.get("confirm"))
            return payload

        with mock.patch.object(chat_commands, "_call_platform_tool", _fake):
            return asyncio.run(chat_commands.handle_chat_command(
                query=query, mode="agent", thinking=False, streaming=False,
                batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
                set_streaming=lambda v: None, set_batch_mode=lambda v: None,
                agent=object(),
            ))

    def test_it_reports_what_was_removed_and_what_survived(self) -> None:
        handled, msg = self._run("/proxy clean wave2d", {
            "status": "ok",
            "removed": ["runs", "optimisation state"],
            "kept": ["sealed references wave2d_ref — shared with suites"],
        })
        self.assertTrue(handled)
        self.assertIn("wave2d", msg)
        self.assertIn("runs", msg)
        # The surviving half is the answer to "I deleted everything and it still
        # remembers" — printing only what was removed would reproduce the confusion.
        self.assertIn("kept", msg)
        self.assertIn("sealed references", msg)

    def test_a_disconnected_proxy_server_says_so(self) -> None:
        handled, msg = self._run("/proxy clean wave2d", None)
        self.assertTrue(handled)
        self.assertIn("not connected", msg)

    def test_usage_without_a_name(self) -> None:
        from mimir.client.ui.cli import chat_commands
        import asyncio
        handled, msg = asyncio.run(chat_commands.handle_chat_command(
            query="/proxy", mode="agent", thinking=False, streaming=False,
            batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
            set_streaming=lambda v: None, set_batch_mode=lambda v: None, agent=object()))
        self.assertTrue(handled)
        self.assertIn("/proxy clean <name>", msg)

    def test_the_command_is_listed_in_help(self) -> None:
        from mimir.client.ui.cli import chat_commands
        import asyncio
        _handled, msg = asyncio.run(chat_commands.handle_chat_command(
            query="/help", mode="agent", thinking=False, streaming=False,
            batch_mode=False, set_mode=lambda v: None, set_thinking=lambda v: None,
            set_streaming=lambda v: None, set_batch_mode=lambda v: None))
        self.assertIn("/proxy clean", msg)


if __name__ == "__main__":
    unittest.main()

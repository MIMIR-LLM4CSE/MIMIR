"""The webview's slash list must name commands the backend actually handles.

Two kinds of slash share the chat input and travel differently: a SKILL is part of the
query and reaches the model, a SESSION COMMAND is handled by `ws_session._handle_command`
and never does. The webview's dropdown listed only skills, and typed session commands
were sent as `type: "query"` — so "/memory clear" asked the model to clear memory rather
than clearing it, and "/proxy clean foo" was a request for the model to tidy up.

The webview now carries its own list (SESSION_COMMANDS in slashUtils.ts), mirrored by
hand so the dropdown needs no round-trip before the first keystroke completes. Mirrored
lists drift; this test is what stops the drift from becoming a command that autocompletes
and then does nothing.
"""
from __future__ import annotations

import os
import re
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SLASH_UTILS = os.path.join(
    _ROOT, "vscode-extension", "webview", "src", "components", "slashUtils.ts")
_WS_SESSION = os.path.join(_ROOT, "client", "ui", "ws", "ws_session.py")
_CLI_COMMANDS = os.path.join(_ROOT, "client", "ui", "cli", "chat_commands.py")


def _advertised() -> list[str]:
    """Command names the webview offers in its dropdown."""
    with open(_SLASH_UTILS, encoding="utf-8") as fh:
        src = fh.read()
    block = src.split("SESSION_COMMANDS", 1)[1].split("];", 1)[0]
    return re.findall(r'name:\s*"([^"]+)"', block)


def _ws_handled() -> str:
    with open(_WS_SESSION, encoding="utf-8") as fh:
        return fh.read()


class AdvertisedCommandsExistTests(unittest.TestCase):
    def test_the_list_is_not_empty(self) -> None:
        # A parser that silently matched nothing would make every assertion below vacuous.
        self.assertGreater(len(_advertised()), 5)

    def test_every_advertised_command_is_handled_by_the_session(self) -> None:
        handler = _ws_handled()
        missing = [
            name for name in _advertised()
            if f'"/{name} "' not in handler and f'"/{name}"' not in handler
        ]
        self.assertEqual(
            missing, [],
            "the webview offers these but ws_session._handle_command does not handle "
            f"them — they would autocomplete and do nothing: {missing}")

    def test_the_two_commands_this_work_added_are_both_there(self) -> None:
        advertised = _advertised()
        self.assertIn("memory", advertised)
        self.assertIn("proxy", advertised)

    def test_they_are_handled_on_the_cli_surface_too(self) -> None:
        # The CLI has its own table; a command added to one surface and not the other is
        # a command that works in one window and not the next.
        with open(_CLI_COMMANDS, encoding="utf-8") as fh:
            cli = fh.read()
        for name in ("memory", "proxy"):
            self.assertIn(f'cmd == "/{name}"', cli, f"/{name} is missing from the CLI")


class RoutingTests(unittest.TestCase):
    """A session command must leave the webview as `command`, not as `query`."""

    def test_the_app_routes_session_commands_apart(self) -> None:
        app = os.path.join(_ROOT, "vscode-extension", "webview", "src", "App.tsx")
        with open(app, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("isSessionCommand(text)", src)
        # The routing check must come BEFORE the busy/steer branch, or a command typed
        # during a run would be queued as a steer and handed to the model.
        self.assertLess(src.index("isSessionCommand(text)"), src.index('type: "steer"'))

    def test_the_dropdown_offers_both_kinds(self) -> None:
        with open(_SLASH_UTILS, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("filterSlashItems", src)
        self.assertIn("SESSION_COMMANDS", src)


if __name__ == "__main__":
    unittest.main()

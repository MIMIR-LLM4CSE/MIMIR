"""The command registry: one grammar, each front-end rendering the same result.

Pure-Python (no agent, no servers): runs on x86 and ARM.
"""
from __future__ import annotations

import asyncio
import unittest

from mimir.client.core.settings import read_setting
from mimir.client.session.commands import (
    COMMANDS, Command, CommandHost, CommandRegistry, CommandResult,
)
from mimir.client.ui.cli.chat_commands import CLI_COMMANDS, render_text


class _Host(CommandHost):
    def __init__(self) -> None:
        self.values: dict = {}

    def apply(self, name, value) -> str:
        self.values[name] = value
        return ""

    def read(self, name):
        return self.values.get(name, read_setting(None, name))


def _run(registry: CommandRegistry, text: str, host: CommandHost | None = None):
    return asyncio.run(registry.run(host or _Host(), text))


class RegistryTests(unittest.TestCase):
    def test_text_that_is_not_a_command_is_left_alone(self) -> None:
        self.assertIsNone(_run(COMMANDS, "explain this function"))

    def test_an_unknown_command_is_an_error_naming_it(self) -> None:
        result = _run(COMMANDS, "/frobnicate now")
        self.assertTrue(result.failed)
        self.assertIn("/frobnicate", result.title)

    def test_names_are_case_insensitive(self) -> None:
        self.assertEqual(COMMANDS.find("/MODE plan")[0].name, "mode")

    def test_two_commands_cannot_answer_to_one_name(self) -> None:
        class _A(Command):
            name = "a"

        class _B(Command):
            name = "b"
            aliases = ("a",)

        with self.assertRaises(ValueError):
            CommandRegistry([_A(), _B()])

    def test_a_surface_offers_only_its_commands(self) -> None:
        ws = COMMANDS.for_surface("ws")
        self.assertIsNone(ws.find("/trust x"))
        self.assertIsNotNone(CLI_COMMANDS.find("/trust x"))

    def test_the_result_carries_the_command_that_made_it(self) -> None:
        self.assertEqual(_run(COMMANDS, "/mode plan").label, "/mode")

    def test_help_names_every_command_of_the_terminal(self) -> None:
        text = _run(CLI_COMMANDS, "/help").title
        for command in CLI_COMMANDS.commands:
            if command.help:
                self.assertIn(f"/{command.name}", text)


class SettingCommandTests(unittest.TestCase):
    def test_a_setting_is_applied_through_the_host(self) -> None:
        host = _Host()
        _run(COMMANDS, "/approvals all", host)
        self.assertEqual(host.values, {"approval_mode": "auto_all"})

    def test_thinking_on_is_auto_and_a_digit_is_the_rung(self) -> None:
        host = _Host()
        _run(COMMANDS, "/thinking on", host)
        self.assertEqual(host.values["thinking_depth"], read_setting(None, "thinking_depth"))
        _run(COMMANDS, "/thinking-depth 0", host)
        self.assertEqual(host.values["thinking_depth"], 0)

    def test_the_former_name_of_a_rung_is_taken_everywhere(self) -> None:
        # The socket refused "parallel" while the terminal took it.
        from unittest import mock
        from mimir.client.session.commands import settings
        host = _Host()
        with mock.patch.object(settings, "save_subagent_level"):
            result = _run(COMMANDS.for_surface("ws"), "/subagents parallel", host)
        self.assertFalse(result.failed)
        self.assertEqual(host.values["subagent_level"], "build")

    def test_a_refusal_from_the_host_is_the_answer(self) -> None:
        host = _Host()
        host.apply = lambda name, value: "Agent is not ready yet."
        result = _run(COMMANDS, "/mode plan", host)
        self.assertTrue(result.failed)
        self.assertIn("not ready", result.title)

    def test_bare_reports_without_changing(self) -> None:
        host = _Host()
        result = _run(COMMANDS, "/context", host)
        self.assertEqual(host.values, {})
        self.assertEqual(result.state, {"type": "context_mode",
                                        "mode": read_setting(None, "context_mode")})


class TextRenderingTests(unittest.TestCase):
    def test_an_error_is_marked(self) -> None:
        self.assertIn("❌ nope", render_text(CommandResult.error("nope")))

    def test_items_are_listed_under_the_title(self) -> None:
        from mimir.client.session.commands import Item
        text = render_text(CommandResult("2 memories", [Item("a", "first"), Item("b")]))
        self.assertIn("2 memories", text)
        self.assertIn("  a — first", text)
        self.assertIn("  b", text)

    def test_nothing_to_say_prints_nothing(self) -> None:
        self.assertEqual(render_text(CommandResult(card=False)), "")


if __name__ == "__main__":
    unittest.main()

"""Tests for the server/skill toggle feature (soft-hide) and its persistence.

The toggle methods on ``MimirAgent`` read only plain instance attributes, so they are
exercised here as unbound methods on a lightweight stand-in — no model or live MCP
server connection is required.
"""

import asyncio
import tempfile
import types
import unittest
from pathlib import Path

import mimir.client.config.preferences as preferences
from mimir.client.agent_core import MimirAgent
from mimir.client.ui.cli.chat_commands import handle_chat_command


def _fake_agent(**overrides):
    """A minimal object carrying just the attributes the toggle methods touch."""
    fake = types.SimpleNamespace(
        tools=[],
        tool_owner={},
        disabled_servers=set(),
        disabled_skills=set(),
        skills={},
    )
    for k, v in overrides.items():
        setattr(fake, k, v)
    return fake


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": ""}}


class PreferencesPersistenceTests(unittest.TestCase):
    def test_round_trip_disabled_sets(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orig = preferences.STATE_DIR
            preferences.STATE_DIR = d
            try:
                preferences.save_disabled({"strings", "datetime"}, {"proxy-optimize"}, {"authz_reminder"})
                servers, skills, nudges = preferences.load_disabled()
                self.assertEqual(servers, {"strings", "datetime"})
                self.assertEqual(skills, {"proxy-optimize"})
                self.assertEqual(nudges, {"authz_reminder"})
                self.assertTrue((Path(d) / "preferences.json").is_file())
            finally:
                preferences.STATE_DIR = orig

    def test_missing_file_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            orig = preferences.STATE_DIR
            preferences.STATE_DIR = d
            try:
                servers, skills, nudges = preferences.load_disabled()
                self.assertEqual(servers, set())
                self.assertEqual(skills, set())
                self.assertEqual(nudges, set())
            finally:
                preferences.STATE_DIR = orig

    def test_corrupt_file_does_not_raise(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "preferences.json").write_text("{not json", encoding="utf-8")
            orig = preferences.STATE_DIR
            preferences.STATE_DIR = d
            try:
                self.assertEqual(preferences.load_disabled(), (set(), set(), set()))
            finally:
                preferences.STATE_DIR = orig


class AdvertisedToolsTests(unittest.TestCase):
    def test_no_disabled_returns_all(self) -> None:
        fake = _fake_agent(
            tools=[_tool("read_file_lines"), _tool("string_op")],
            tool_owner={"read_file_lines": "files", "string_op": "strings"},
        )
        out = MimirAgent.advertised_tools(fake)
        self.assertEqual([t["function"]["name"] for t in out], ["read_file_lines", "string_op"])

    def test_disabled_server_tools_hidden(self) -> None:
        fake = _fake_agent(
            tools=[_tool("read_file_lines"), _tool("string_op"), _tool("date_op")],
            tool_owner={"read_file_lines": "files", "string_op": "strings", "date_op": "datetime"},
            disabled_servers={"strings", "datetime"},
        )
        out = [t["function"]["name"] for t in MimirAgent.advertised_tools(fake)]
        self.assertEqual(out, ["read_file_lines"])

    def test_unknown_owner_is_kept_fail_open(self) -> None:
        fake = _fake_agent(
            tools=[_tool("mystery")],
            tool_owner={},  # no owner recorded
            disabled_servers={"strings"},
        )
        out = [t["function"]["name"] for t in MimirAgent.advertised_tools(fake)]
        self.assertEqual(out, ["mystery"])


class SkillToggleTests(unittest.TestCase):
    def test_skill_enabled_default(self) -> None:
        fake = _fake_agent()
        self.assertTrue(MimirAgent.skill_enabled(fake, "fix-bug"))

    def test_skill_disabled(self) -> None:
        fake = _fake_agent(disabled_skills={"fix-bug"})
        self.assertFalse(MimirAgent.skill_enabled(fake, "fix-bug"))


class TogglesStateTests(unittest.TestCase):
    def test_state_shape_and_enabled_flags(self) -> None:
        fake = _fake_agent(
            disabled_servers={"strings"},
            disabled_skills={"finetune"},
            skills={"fix-bug": {"description": "Fix a bug"},
                    "finetune": {"description": "Train a model"}},
        )
        state = MimirAgent.toggles_state(fake)
        self.assertIn("servers", state)
        self.assertIn("skills", state)

        servers = {s["name"]: s for s in state["servers"]}
        # every server from the static registry is listed
        self.assertIn("files", servers)
        self.assertTrue(servers["files"]["enabled"])
        self.assertFalse(servers["strings"]["enabled"])
        self.assertTrue(servers["files"]["description"])  # tooltip text present

        skills = {s["name"]: s for s in state["skills"]}
        self.assertTrue(skills["fix-bug"]["enabled"])
        self.assertFalse(skills["finetune"]["enabled"])


class SetEnabledPersistsTests(unittest.TestCase):
    def test_set_server_enabled_mutates_and_saves(self) -> None:
        saved = {}

        def fake_save(servers, skills):
            saved["servers"] = set(servers)
            saved["skills"] = set(skills)

        fake = _fake_agent()
        fake._save_toggles = lambda: fake_save(fake.disabled_servers, fake.disabled_skills)

        MimirAgent.set_server_enabled(fake, "strings", False)
        self.assertIn("strings", fake.disabled_servers)
        self.assertEqual(saved["servers"], {"strings"})

        MimirAgent.set_server_enabled(fake, "strings", True)
        self.assertNotIn("strings", fake.disabled_servers)
        self.assertEqual(saved["servers"], set())

    def test_set_skill_enabled_mutates_and_saves(self) -> None:
        fake = _fake_agent()
        fake._save_toggles = lambda: None
        MimirAgent.set_skill_enabled(fake, "fix-bug", False)
        self.assertIn("fix-bug", fake.disabled_skills)
        MimirAgent.set_skill_enabled(fake, "fix-bug", True)
        self.assertNotIn("fix-bug", fake.disabled_skills)


class ChatCommandTests(unittest.TestCase):
    """The `/servers` and `/skills` CLI commands list and toggle via the agent."""

    def _run(self, query, agent):
        async def _noop(*a, **k):
            return None
        return asyncio.run(handle_chat_command(
            query=query, mode="agent", thinking=False, streaming=False, batch_mode=False,
            set_mode=lambda m: None, set_thinking=lambda b: None, set_streaming=lambda b: None,
            set_batch_mode=lambda b: None, agent=agent,
        ))

    def _agent(self):
        calls = []
        return types.SimpleNamespace(
            toggles_state=lambda: {
                "servers": [{"name": "files", "description": "Files", "enabled": True},
                            {"name": "strings", "description": "Strings", "enabled": False}],
                "skills": [{"name": "fix-bug", "description": "Fix", "enabled": True}],
            },
            set_server_enabled=lambda n, e: calls.append(("server", n, e)),
            set_skill_enabled=lambda n, e: calls.append(("skill", n, e)),
            _calls=calls,
        )

    def test_servers_list(self):
        handled, msg = self._run("/servers", self._agent())
        self.assertTrue(handled)
        self.assertIn("files", msg)
        self.assertIn("strings", msg)

    def test_servers_toggle_off(self):
        agent = self._agent()
        handled, msg = self._run("/servers off strings", agent)
        self.assertTrue(handled)
        self.assertIn(("server", "strings", False), agent._calls)

    def test_servers_toggle_unknown(self):
        agent = self._agent()
        handled, msg = self._run("/servers on nope", agent)
        self.assertIn("Unknown", msg)
        self.assertEqual(agent._calls, [])

    def test_skills_toggle_on(self):
        agent = self._agent()
        self._run("/skills on fix-bug", agent)
        self.assertIn(("skill", "fix-bug", True), agent._calls)

    def test_unavailable_without_agent(self):
        handled, msg = self._run("/servers", None)
        self.assertTrue(handled)
        self.assertIn("not available", msg)


if __name__ == "__main__":
    unittest.main()

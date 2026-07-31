"""Coverage for centralizing user extensions under ``.mimir/`` — skills + MCP servers.

Exercises `extensions/servers.py` (server discovery + merged registry), the
`load_skills(merge=…)` layering with asymmetric collision, and the merged toggle surface.
Nothing is seeded into ``.mimir/`` — that directory is the user's alone.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import mimir.client.extensions as ur
from mimir.client.config.constants import SERVERS_DIR_ENV, SKILLS_DIR_ENV


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class DiscoverUserServersTests(unittest.TestCase):
    def test_discovers_and_normalizes_names(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "server_weather.py"), "# server\n")
            _write(os.path.join(d, "radar.js"), "// server\n")
            _write(os.path.join(d, "_helper.py"), "x = 1\n")   # skipped (private)
            _write(os.path.join(d, "notes.txt"), "nope\n")      # skipped (not py/js)
            found = ur.discover_user_servers(d)
        self.assertEqual(found.keys(), {"weather", "radar"})
        self.assertTrue(found["weather"].endswith("server_weather.py"))

    def test_collision_with_bundled_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "server_files.py"), "# shadow attempt\n")  # 'files' is bundled
            with self.assertLogs("mimir.client.extensions.servers", level="WARNING"):
                found = ur.discover_user_servers(d)
        self.assertNotIn("files", found)

    def test_absent_dir_is_noop(self) -> None:
        self.assertEqual(ur.discover_user_servers("/no/such/dir"), {})

    def test_all_servers_merges_and_keeps_core(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "server_demo.py"), "# server\n")
            with mock.patch.dict(os.environ, {SERVERS_DIR_ENV: d}):
                servers = ur.all_servers()
                descs = ur.all_server_descriptions()
        self.assertIn("demo", servers)
        self.assertIn("files", servers)          # bundled core preserved
        self.assertIn("demo", descs)
        self.assertIn(".mimir/servers", descs["demo"])

    def test_resolvers_honor_env(self) -> None:
        with mock.patch.dict(os.environ, {SKILLS_DIR_ENV: "/tmp/xskills", SERVERS_DIR_ENV: "/tmp/xservers"}):
            self.assertEqual(ur.resolve_skills_dir(), "/tmp/xskills")
            self.assertEqual(ur.resolve_servers_dir(), "/tmp/xservers")


_SKILL_MD = """\
---
name: {name}
description: {desc}
---

Body for {name}.
"""


class UserSkillLoadingTests(unittest.TestCase):
    def _agent(self):
        # Import lazily so a construction failure surfaces here, not at collection.
        from mimir.client.agent_core import MimirAgent
        from mimir.client.config import DEFAULT_MODEL
        return MimirAgent(model=DEFAULT_MODEL)

    def test_user_skill_added_and_overrides_bundled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            # A brand-new user skill…
            _write(os.path.join(d, "mytask", "SKILL.md"), _SKILL_MD.format(name="mytask", desc="my custom skill"))
            # …and an override of a bundled one (fix-bug ships in mimir/skills).
            _write(os.path.join(d, "fix-bug", "SKILL.md"), _SKILL_MD.format(name="fix-bug", desc="OVERRIDDEN"))
            with mock.patch.dict(os.environ, {SKILLS_DIR_ENV: d}):
                agent = self._agent()
        self.assertIn("mytask", agent.skills)
        self.assertEqual(agent.skills["mytask"]["description"], "my custom skill")
        self.assertEqual(agent.skills["fix-bug"]["description"], "OVERRIDDEN")  # user wins
        self.assertIn("write-tests", agent.skills)  # other bundled skills intact

    def test_user_server_appears_in_toggles(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            _write(os.path.join(d, "server_demo.py"), "# server\n")
            with mock.patch.dict(os.environ, {SERVERS_DIR_ENV: d}):
                agent = self._agent()
                state = agent.toggles_state()
        names = {row["name"] for row in state["servers"]}
        self.assertIn("demo", names)
        self.assertIn("files", names)  # bundled still listed
        demo = next(r for r in state["servers"] if r["name"] == "demo")
        self.assertTrue(demo["enabled"])
        self.assertTrue(demo["description"])


if __name__ == "__main__":
    unittest.main()

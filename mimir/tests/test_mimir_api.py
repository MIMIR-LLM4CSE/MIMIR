"""MIMIR's self-knowledge has to be the running build's, not a remembered one.

The ``mimir_api`` server exists because a prose copy of the extension API drifts: the
documentation already claimed a flag count the vocabulary had outgrown. So the tests
that matter here are the parity ones — every capability the client defines is reported,
every flag reported is a real constant, and the drop-in path handed to a user is the one
the loader will scan. The rest covers the shapes a caller depends on.
"""

import importlib.util
import os
import pathlib
import sys
import unittest
from unittest import mock

SERVERS_DIR = pathlib.Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "agent_state"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from mimir.client.config import constants as k                      # noqa: E402
from mimir.client.context import capabilities as caps               # noqa: E402
from mimir.client.extensions import resolve_skills_dir              # noqa: E402
from mimir.servers._shared import extension_paths                   # noqa: E402


def _load():
    path = SERVERS_DIR / "agent_state" / "server_mimir_api.py"
    spec = importlib.util.spec_from_file_location("server_mimir_api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


api = _load()

#: The declared capability vocabulary, as the client publishes it.
_CLIENT_FLAGS = {
    getattr(caps, name) for name in caps.__all__
    if name.isupper() and isinstance(getattr(caps, name), str)
}


class CapabilityVocabularyTests(unittest.TestCase):
    """The flag list is parsed from the live module, so it cannot fall behind it."""

    @classmethod
    def setUpClass(cls):
        cls.payload = api.mimir_api("capabilities")

    def test_every_published_capability_is_reported(self):
        reported = {flag["value"] for flag in self.payload["flags"]}
        self.assertTrue(_CLIENT_FLAGS <= reported, _CLIENT_FLAGS - reported)

    def test_every_reported_flag_is_a_real_constant(self):
        for flag in self.payload["flags"]:
            with self.subTest(flag=flag["name"]):
                self.assertEqual(getattr(caps, flag["name"]), flag["value"])

    def test_every_flag_says_what_it_drives(self):
        # A flag with no note is a flag nobody can choose between.
        for flag in self.payload["flags"]:
            with self.subTest(flag=flag["name"]):
                self.assertTrue(flag["note"].strip())
                self.assertTrue(flag["group"].strip())

    def test_reversibility_is_reported_apart_from_the_flags(self):
        levels = [level["value"] for level in self.payload["reversibility"]]
        self.assertEqual(sorted(levels), sorted(caps.REVERSIBILITY_LEVELS))
        self.assertFalse({flag["value"] for flag in self.payload["flags"]}
                         & set(caps.REVERSIBILITY_LEVELS))

    def test_a_derived_flag_is_marked_undeclarable(self):
        # Declaring SENSITIVE says nothing: the client derives it from reversibility.
        sensitive = next(f for f in self.payload["flags"] if f["value"] == caps.SENSITIVE)
        self.assertIs(sensitive["declarable"], False)
        declarable = [f for f in self.payload["flags"] if f.get("declarable") is not False]
        self.assertEqual(self.payload["count"], len(self.payload["flags"]))
        self.assertTrue(declarable)


class DropInPathTests(unittest.TestCase):
    """The path a user is told to write is the path the loader reads."""

    def test_paths_resolve_under_the_workspace_mimir_dir(self):
        with mock.patch.dict(os.environ, {"MCP_FILES_ROOT": "/tmp/ws-under-test"},
                             clear=False):
            for var in (k.SKILLS_DIR_ENV, k.SERVERS_DIR_ENV, k.PLUGINS_DIR_ENV,
                        k.SYSTEM_PROMPT_ENV):
                os.environ.pop(var, None)
            index = api.mimir_api("index")
            self.assertEqual(index["mimir_dir"], "/tmp/ws-under-test/.mimir")
            for entry in index["extension_types"]:
                with self.subTest(type=entry["type"]):
                    self.assertTrue(entry["drop_in"].startswith("/tmp/ws-under-test/.mimir"))
                    self.assertIsNone(entry["env_override"]["set_to"])

    def test_the_skill_path_matches_what_the_client_loader_scans(self):
        # Same question asked of both ends; one definition, so one answer. The client
        # pins its workspace root at import and the server resolves it per call, so the
        # premise of the agreement is the one the real stack arranges: both ends see the
        # same MCP_FILES_ROOT.
        with mock.patch.dict(os.environ, {"MCP_FILES_ROOT": k.WORKSPACE_ROOT},
                             clear=False):
            os.environ.pop(k.SKILLS_DIR_ENV, None)
            reported = api.mimir_api("skill")["drop_in"]
            self.assertEqual(os.path.dirname(os.path.dirname(reported)),
                             resolve_skills_dir())

    def test_an_env_override_is_reported_as_the_truth(self):
        with mock.patch.dict(os.environ, {k.SKILLS_DIR_ENV: "/tmp/elsewhere/skills"}):
            entry = api.mimir_api("skill")
            self.assertEqual(entry["drop_in"], "/tmp/elsewhere/skills/<name>/SKILL.md")
            self.assertEqual(entry["env_override"]["set_to"], "/tmp/elsewhere/skills")

    def test_the_client_re_exports_one_definition_of_the_names(self):
        for name in ("SKILLS_DIR_ENV", "SKILLS_DIRNAME", "SERVERS_DIR_ENV",
                     "SERVERS_DIRNAME", "PLUGINS_DIR_ENV", "PLUGINS_DIRNAME",
                     "SYSTEM_PROMPT_ENV", "SYSTEM_PROMPT_FILENAME"):
            with self.subTest(name=name):
                self.assertEqual(getattr(k, name), getattr(extension_paths, name))


class TopicTests(unittest.TestCase):
    def test_the_index_covers_every_type_and_names_its_topic(self):
        index = api.mimir_api("index")
        listed = {entry["type"] for entry in index["extension_types"]}
        self.assertEqual(listed, set(api._TYPES))
        self.assertTrue(listed <= set(index["topics"]))
        self.assertIn("index", index["topics"])

    def test_every_type_topic_ships_a_readable_template(self):
        for kind in api._TYPES:
            with self.subTest(type=kind):
                entry = api.mimir_api(kind)
                template = entry["template"]
                self.assertTrue(os.path.isfile(template["path"]), template["path"])
                self.assertNotIn("[unavailable", template["source"])
                self.assertTrue(template["source"].strip())
                self.assertTrue(entry["authoring_rules"])

    def test_the_default_topic_is_the_index(self):
        self.assertEqual(api.mimir_api()["extension_types"],
                         api.mimir_api("index")["extension_types"])
        self.assertEqual(api.mimir_api("post-tool")["type"], "post_tool")

    def test_an_unknown_topic_names_the_ones_that_exist(self):
        payload = api.mimir_api("plugins")
        self.assertEqual(payload["status"], "error")
        for topic in api._TOPICS:
            self.assertIn(topic, payload["hint"])

    def test_reserved_namespaces_come_from_the_registry(self):
        # A filename is not a namespace: server_spawn_agent.py registers as `agent`.
        reserved = api.mimir_api("server")["reserved_server_namespaces"]
        self.assertEqual(set(reserved["names"]), set(k.SERVERS))
        self.assertIn("agent", reserved["names"])
        self.assertNotIn("spawn_agent", reserved["names"])

    def test_loaded_reports_what_the_workspace_actually_has(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                (root / ".mimir" / "skills" / "mine").mkdir(parents=True)
                (root / ".mimir" / "skills" / "mine" / "SKILL.md").write_text("---\n")
                (root / ".mimir" / "skills" / "halfway").mkdir()      # no SKILL.md
                (root / ".mimir" / "plugins").mkdir()
                (root / ".mimir" / "plugins" / "guard.py").write_text("")
                (root / ".mimir" / "servers").mkdir()
                (root / ".mimir" / "servers" / "server_demo.py").write_text("")
                (root / ".mimir" / "servers" / "_helper.py").write_text("")
                os.environ["MCP_FILES_ROOT"] = str(root)
                for var in (k.SKILLS_DIR_ENV, k.SERVERS_DIR_ENV, k.PLUGINS_DIR_ENV,
                            k.SYSTEM_PROMPT_ENV):
                    os.environ.pop(var, None)
                payload = api.mimir_api("loaded")
        user = payload["user"]
        self.assertEqual(user["skills"]["present"], ["mine"])     # not the empty dir
        self.assertEqual(user["plugins"]["present"], ["guard.py"])
        self.assertEqual(user["servers"]["present"], ["server_demo.py"])  # not _helper
        self.assertFalse(user["system_prompt"]["present"])

    def test_bundled_skills_are_reported_with_their_descriptions(self):
        bundled = {skill["name"]: skill for skill in api.mimir_api("loaded")["bundled_skills"]}
        self.assertIn("mimir-api", bundled)
        self.assertTrue(bundled["mimir-api"]["description"])


class BundledSkillContractTests(unittest.TestCase):
    """The contract the loader enforces silently: front-matter name == directory."""

    def test_every_bundled_skill_names_its_own_directory(self):
        # A mismatch costs the skill its slash command with no error anywhere, so it is
        # checked against the directories on disk rather than against what the API
        # reports (which reads the front-matter name).
        for directory in sorted(pathlib.Path(k.SKILL_BASE).iterdir()):
            md = directory / "SKILL.md"
            if not md.is_file():
                continue
            with self.subTest(skill=directory.name):
                front = api._front_matter(str(md))
                self.assertEqual(front.get("name"), directory.name)
                self.assertTrue(front.get("description"))

    def test_the_mimir_api_skill_points_at_the_tool(self):
        path = pathlib.Path(k.SKILL_BASE) / "mimir-api" / "SKILL.md"
        body = path.read_text()
        self.assertIn("name: mimir-api", body)
        self.assertIn("mimir_api(", body)


if __name__ == "__main__":
    unittest.main()

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


class LoadSkillTests(unittest.TestCase):
    """The pull itself: what the model gets, and what it is refused.

    The body must come back byte-for-byte. It is a methodology the model is about to
    follow, so a server that reflowed or trimmed it would be changing the instruction
    while appearing to serve it.
    """

    def test_the_body_is_the_file_verbatim(self):
        from mimir.client.agent_core import _parse_skill_markdown
        source = (pathlib.Path(k.SKILL_BASE) / "fix-bug" / "SKILL.md").read_text()
        result = api.load_skill("fix-bug")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["skill"], "fix-bug")
        # Parity with the client's own parser, which is what the user's /<name> path
        # folds into messages[0]: the two routes must hand over the same text.
        self.assertEqual(result["instructions"],
                         _parse_skill_markdown(source)["content"])
        self.assertTrue(result["description"])

    def test_a_user_invoked_only_skill_is_refused_with_its_slash_command(self):
        # prepare-pr ships with disable-model-invocation: true. Before that field was
        # honoured it was parsed and dropped, so the skill was reachable by the model
        # against its own declaration.
        result = api.load_skill("prepare-pr")
        self.assertEqual(result["status"], "error")
        self.assertIn("user-invoked only", result["error"])
        self.assertIn("/prepare-pr", result["hint"])

    def test_an_unknown_name_is_refused_with_the_available_ones(self):
        result = api.load_skill("no-such-skill")
        self.assertEqual(result["status"], "error")
        self.assertIn("No skill named", result["error"])
        self.assertIn("fix-bug", result["hint"])

    def test_a_name_that_could_leave_its_directory_is_never_joined(self):
        # The one path-traversal surface this server adds: `name` arrives from the
        # model and is joined onto the skills directories.
        for name in ("../../etc/passwd", "a/b", ".hidden", "", "..",
                     "fix-bug/../../../etc/passwd"):
            with self.subTest(name=name):
                result = api.load_skill(name)
                self.assertEqual(result["status"], "error")

    def test_a_user_skill_overrides_the_bundled_one_of_the_same_name(self):
        import tempfile
        from mimir.client.agent_core import MimirAgent
        with tempfile.TemporaryDirectory() as d:
            skill_dir = pathlib.Path(d) / "fix-bug"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: fix-bug\ndescription: OVERRIDDEN\n---\nMy own method.\n"
            )
            with mock.patch.dict(os.environ, {extension_paths.SKILLS_DIR_ENV: d}):
                result = api.load_skill("fix-bug")
                self.assertEqual(result["instructions"], "My own method.")
                self.assertEqual(result["description"], "OVERRIDDEN")
                # And the client's loader agrees: the override rule is restated across
                # a process boundary, so parity is tested rather than trusted.
                agent = MimirAgent.__new__(MimirAgent)
                agent.skills = {}
                agent.load_skills(k.SKILL_BASE)
                agent.load_skills(resolve_skills_dir(), merge=True)
                self.assertEqual(agent.skills["fix-bug"]["content"], "My own method.")

    def test_the_inventory_reports_which_skills_the_model_may_load(self):
        by_name = {entry["name"]: entry for entry in api._bundled_skill_names()}
        self.assertFalse(by_name["prepare-pr"]["model_invocable"])
        self.assertTrue(by_name["fix-bug"]["model_invocable"])


if __name__ == "__main__":
    unittest.main()

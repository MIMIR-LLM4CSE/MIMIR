"""Tests for the memory server: duplicate refusal, and the two memory scopes.

A near-repeat of ANY stored memory is refused, not only of the recent ones, and the
refusal is an error naming the memory to update — a silent "ok" let the model believe
it had stored a second copy.

The scope tests pin what separates the two stores: a global fact reaches every
workspace, so it blocks a local copy of itself, is never aged out to make room, and
survives a workspace wipe.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from _memory_fixtures import load_memory_server, point_stores


class _MemoryTestCase(unittest.TestCase):
    """Both stores under one tmpdir, and no embedding backend."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.mem = load_memory_server()
        self.ws_dir, self.global_dir = point_stores(self.mem, tmp.name)
        # Keep the suite hermetic: no embedding backend.
        original = self.mem._embed.is_available
        self.mem._embed.is_available = lambda: False
        self.addCleanup(setattr, self.mem._embed, "is_available", original)


class MemoryDuplicateTests(_MemoryTestCase):
    def _fill(self):
        first = self.mem.memory_add(
            "The project runs its tests with pytest from the mimir directory",
            scope="workspace",
            description="tests run with pytest",
        )
        for i, topic in enumerate(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]):
            self.mem.memory_add(f"Unrelated fact {i} about the {topic} subsystem layout",
                                scope="workspace", description=f"{topic} layout")
        return first["name"]

    def test_a_repeat_of_the_oldest_memory_is_refused_with_its_name(self):
        oldest = self._fill()
        res = self.mem.memory_add(
            "The project runs its tests with pytest from the mimir directory.",
            scope="workspace",
        )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["similar_memory"], oldest)
        self.assertIn(oldest, res["error"])
        self.assertIn("Update", res["hint"])
        self.assertEqual(self.mem.memory_list_all()["count"], 7)

    def test_a_distinct_fact_is_stored(self):
        self._fill()
        res = self.mem.memory_add("GPU jobs go to the a100 partition with a two hour limit",
                                  scope="workspace")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self.mem.memory_list_all()["count"], 8)

    def test_update_keeps_the_file_name(self):
        name = self._fill()
        res = self.mem.memory_update(name, text="Tests run with pytest -q from the repo root")
        self.assertEqual(res["status"], "ok")
        self.assertTrue(os.path.exists(os.path.join(self.mem.MEMORY_DIR, f"{name}.md")))
        with open(self.mem.INDEX_FILE, encoding="utf-8") as f:
            self.assertIn(f"({name}.md)", f.read())


class MemoryScopeTests(_MemoryTestCase):
    """What the global scope is for, and what keeps the two stores apart."""

    def test_scope_is_required_and_the_error_says_how_to_choose(self):
        res = self.mem.memory_add("The user prefers French for explanations")
        self.assertEqual(res["status"], "error")
        self.assertIn("scope is required", res["error"])
        self.assertIn("workspace", res["hint"])
        self.assertIn("global", res["hint"])
        # Nothing written anywhere.
        self.assertEqual(self.mem.memory_list_all()["count"], 0)

    def test_an_unknown_scope_is_refused_naming_the_valid_values(self):
        res = self.mem.memory_add("something", scope="everywhere")
        self.assertEqual(res["status"], "error")
        self.assertIn("everywhere", res["error"])
        self.assertIn("workspace", res["hint"])

    def test_a_global_memory_lands_in_the_global_store_and_says_so(self):
        res = self.mem.memory_add("The user wants commits only when they ask for one",
                                  scope="global", description="commit only when asked")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["scope"], "global")
        self.assertTrue(res["path"].startswith(str(self.global_dir)))
        self.assertTrue(os.path.exists(res["path"]))
        # And nothing in the workspace store.
        self.assertEqual(self.mem.memory_list_all(scope="workspace")["count"], 0)

    def test_search_spans_both_stores_and_tags_each_result(self):
        self.mem.memory_add("The user wants commits only when they ask for one",
                            scope="global", description="commit only when asked")
        self.mem.memory_add("This project builds its wheel with pyproject and hatchling",
                            scope="workspace", description="build backend")
        res = self.mem.memory_search("commits", limit=5)
        self.assertEqual(res["status"], "ok")
        self.assertEqual([r["scope"] for r in res["results"]], ["global"])

        everything = self.mem.memory_list_all()
        self.assertEqual({e["scope"] for e in everything["memory"]}, {"global", "workspace"})
        only_ws = self.mem.memory_list_all(scope="workspace")
        self.assertEqual([e["scope"] for e in only_ws["memory"]], ["workspace"])

    def test_a_global_fact_blocks_a_workspace_copy_and_names_the_scope(self):
        self.mem.memory_add("The user wants commits only when they explicitly ask for one",
                            scope="global", description="commit only when asked")
        res = self.mem.memory_add(
            "The user wants commits only when they explicitly ask for one",
            scope="workspace",
        )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["similar_scope"], "global")
        self.assertIn("global", res["error"])

    def test_a_workspace_fact_does_not_block_a_global_add(self):
        # The asymmetry is the point: one project's note must never stand in the way
        # of a preference that holds everywhere.
        self.mem.memory_add("The user wants commits only when they explicitly ask for one",
                            scope="workspace", description="commit only when asked")
        res = self.mem.memory_add(
            "The user wants commits only when they explicitly ask for one",
            scope="global",
        )
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["scope"], "global")

    def test_delete_without_a_scope_finds_the_global_one(self):
        added = self.mem.memory_add("The user writes in French", scope="global",
                                    description="user writes French")
        self.mem.memory_add("This project targets Python 3.11", scope="workspace",
                            description="python version")
        res = self.mem.memory_delete(added["name"])
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["scope"], "global")
        self.assertFalse(os.path.exists(added["path"]))
        self.assertEqual(self.mem.memory_list_all(scope="workspace")["count"], 1)

    def test_a_name_in_both_scopes_is_an_error_until_a_scope_is_given(self):
        a = self.mem.memory_add("Alpha fact about the build", scope="workspace",
                                description="shared slug")
        b = self.mem.memory_add("Beta fact about the user's editor", scope="global",
                                description="shared slug")
        self.assertEqual(a["name"], b["name"])  # slugs are unique per store, not across

        res = self.mem.memory_delete(a["name"])
        self.assertEqual(res["status"], "error")
        self.assertEqual(sorted(res["scopes"]), ["global", "workspace"])

        res = self.mem.memory_delete(a["name"], scope="workspace")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["scope"], "workspace")
        self.assertTrue(os.path.exists(b["path"]))

    def test_clear_wipes_only_the_workspace_by_default(self):
        kept = self.mem.memory_add("The user wants commits only when they ask",
                                   scope="global", description="commit only when asked")
        self.mem.memory_add("This project uses hatchling", scope="workspace",
                            description="build backend")
        res = self.mem.memory_clear()
        self.assertEqual(res["status"], "ok")
        self.assertEqual((res["scope"], res["cleared"]), ("workspace", 1))
        self.assertTrue(os.path.exists(kept["path"]))
        self.assertEqual(self.mem.memory_list_all(scope="global")["count"], 1)

    def test_clear_refuses_all(self):
        self.mem.memory_add("The user wants commits only when they ask", scope="global",
                            description="commit only when asked")
        res = self.mem.memory_clear(scope="all")
        self.assertEqual(res["status"], "error")
        self.assertEqual(self.mem.memory_list_all(scope="global")["count"], 1)

    def test_the_global_store_refuses_at_its_cap_while_the_workspace_prunes(self):
        # Texts share no words: the duplicate refusal is a different rule, tested above.
        for i in range(self.mem._MAX_GLOBAL_ENTRIES):
            res = self.mem.memory_add(f"alpha{i} beta{i} gamma{i} delta{i}",
                                      scope="global", description=f"pref {i}")
            self.assertEqual(res["status"], "ok", res.get("error"))
        full = self.mem.memory_add("epsilon zeta eta theta",
                                   scope="global", description="overflow")
        self.assertEqual(full["status"], "error")
        self.assertIn("full", full["error"])
        self.assertEqual(self.mem.memory_list_all(scope="global")["count"],
                         self.mem._MAX_GLOBAL_ENTRIES)

        # The workspace store still ages out its oldest, and the global one is untouched.
        self.mem._MAX_ENTRIES = 3
        for i in range(5):
            self.mem.memory_add(f"iota{i} kappa{i} lambda{i} mu{i}",
                                scope="workspace", description=f"note {i}")
        self.assertEqual(self.mem.memory_list_all(scope="workspace")["count"], 3)
        self.assertEqual(self.mem.memory_list_all(scope="global")["count"],
                         self.mem._MAX_GLOBAL_ENTRIES)

    def test_each_scope_keeps_its_own_index(self):
        self.mem.memory_add("The user writes in French", scope="global",
                            description="user writes French")
        self.mem.memory_add("This project targets Python 3.11", scope="workspace",
                            description="python version")
        with open(self.mem.GLOBAL_INDEX_FILE, encoding="utf-8") as f:
            global_index = f.read()
        with open(self.mem.INDEX_FILE, encoding="utf-8") as f:
            ws_index = f.read()
        self.assertIn("user-writes-french.md", global_index)
        self.assertNotIn("python-version.md", global_index)
        self.assertIn("python-version.md", ws_index)
        self.assertNotIn("user-writes-french.md", ws_index)

    def test_the_resource_carries_both_indexes_under_their_headings(self):
        self.assertEqual(self.mem.memory_all(), "(no memories stored)")
        self.mem.memory_add("The user writes in French", scope="global",
                            description="user writes French")
        text = self.mem.memory_all()
        self.assertIn("## Global memory", text)
        self.assertNotIn("## Workspace memory", text)  # empty scope is left out
        self.mem.memory_add("This project targets Python 3.11", scope="workspace",
                            description="python version")
        text = self.mem.memory_all()
        self.assertIn("## Global memory", text)
        self.assertIn("## Workspace memory", text)

    def test_nothing_materialises_a_store_directory_before_a_write(self):
        self.assertEqual(self.mem.memory_list_all()["count"], 0)
        self.assertEqual(self.mem.memory_search("anything")["count"], 0)
        self.mem.memory_all()
        self.assertFalse(os.path.isdir(self.ws_dir))
        self.assertFalse(os.path.isdir(self.global_dir))


class MemoryScopedEmbeddingTests(unittest.TestCase):
    """A cross-scope search must not write one scope's vectors into the other's cache."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.mem = load_memory_server()
        self.ws_dir, self.global_dir = point_stores(self.mem, tmp.name)
        self.mem._embed.is_available = lambda: True
        self.mem._embed.embed_model_id = lambda: "fake-model"
        self.mem._embed.embed_texts = lambda texts: [self._vec(t) for t in texts]
        self.mem._embed.embed_one = lambda t: self._vec(t)

    @staticmethod
    def _vec(text: str):
        words = set(text.lower().split())
        return [1.0 if w in words else 0.0 for w in ("french", "python", "commit")]

    def test_each_scope_backfills_only_its_own_cache(self):
        self.mem.memory_add("The user writes in french", scope="global",
                            description="user writes french")
        self.mem.memory_add("This project targets python", scope="workspace",
                            description="python version")

        with open(self.mem.GLOBAL_EMBEDDINGS_FILE, encoding="utf-8") as f:
            global_cache = json.load(f)
        with open(self.mem.EMBEDDINGS_FILE, encoding="utf-8") as f:
            ws_cache = json.load(f)
        self.assertEqual(set(global_cache), {"user-writes-french"})
        self.assertEqual(set(ws_cache), {"python-version"})

        # A search spanning both stores must leave each cache as it found it.
        before = (json.dumps(global_cache, sort_keys=True), json.dumps(ws_cache, sort_keys=True))
        res = self.mem.memory_search("french", limit=2)
        self.assertEqual(res["results"][0]["scope"], "global")
        with open(self.mem.GLOBAL_EMBEDDINGS_FILE, encoding="utf-8") as f:
            after_global = json.dumps(json.load(f), sort_keys=True)
        with open(self.mem.EMBEDDINGS_FILE, encoding="utf-8") as f:
            after_ws = json.dumps(json.load(f), sort_keys=True)
        self.assertEqual(before, (after_global, after_ws))

    def test_a_missing_vector_is_backfilled_into_its_own_scope(self):
        self.mem.memory_add("The user writes in french", scope="global",
                            description="user writes french")
        self.mem.memory_add("This project targets python", scope="workspace",
                            description="python version")
        # Drop the workspace cache: the next search must rebuild it there, and only there.
        os.remove(self.mem.EMBEDDINGS_FILE)
        self.mem.memory_search("python", limit=2)
        with open(self.mem.EMBEDDINGS_FILE, encoding="utf-8") as f:
            self.assertEqual(set(json.load(f)), {"python-version"})
        with open(self.mem.GLOBAL_EMBEDDINGS_FILE, encoding="utf-8") as f:
            self.assertEqual(set(json.load(f)), {"user-writes-french"})


if __name__ == "__main__":
    unittest.main()

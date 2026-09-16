"""Tests for the memory server's duplicate refusal.

A near-repeat of ANY stored memory is refused, not only of the recent ones, and the
refusal is an error naming the memory to update — a silent "ok" let the model believe
it had stored a second copy.
"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SHARED = _ROOT / "mimir" / "servers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))


def _load_server_memory():
    spec = importlib.util.spec_from_file_location(
        "server_memory",
        _ROOT / "mimir" / "servers" / "agent_state" / "server_memory.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryDuplicateTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.mem = _load_server_memory()
        d = Path(tmp.name)
        self.mem.MEMORY_DIR = str(d)
        self.mem.INDEX_FILE = str(d / "MEMORY.md")
        self.mem.EMBEDDINGS_FILE = str(d / "embeddings.json")
        # Keep the suite hermetic: no embedding backend.
        original = self.mem._embed.is_available
        self.mem._embed.is_available = lambda: False
        self.addCleanup(setattr, self.mem._embed, "is_available", original)

    def _fill(self):
        first = self.mem.memory_add(
            "The project runs its tests with pytest from the mimir directory",
            description="tests run with pytest",
        )
        for i, topic in enumerate(["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]):
            self.mem.memory_add(f"Unrelated fact {i} about the {topic} subsystem layout",
                                description=f"{topic} layout")
        return first["name"]

    def test_a_repeat_of_the_oldest_memory_is_refused_with_its_name(self):
        oldest = self._fill()
        res = self.mem.memory_add(
            "The project runs its tests with pytest from the mimir directory.",
        )
        self.assertEqual(res["status"], "error")
        self.assertEqual(res["similar_memory"], oldest)
        self.assertIn(oldest, res["error"])
        self.assertIn("Update", res["hint"])
        self.assertEqual(self.mem.memory_list_all()["count"], 7)

    def test_a_distinct_fact_is_stored(self):
        self._fill()
        res = self.mem.memory_add("GPU jobs go to the a100 partition with a two hour limit")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self.mem.memory_list_all()["count"], 8)

    def test_update_keeps_the_file_name(self):
        name = self._fill()
        res = self.mem.memory_update(name, text="Tests run with pytest -q from the repo root")
        self.assertEqual(res["status"], "ok")
        self.assertTrue(os.path.exists(os.path.join(self.mem.MEMORY_DIR, f"{name}.md")))
        with open(self.mem.INDEX_FILE, encoding="utf-8") as f:
            self.assertIn(f"({name}.md)", f.read())


if __name__ == "__main__":
    unittest.main()

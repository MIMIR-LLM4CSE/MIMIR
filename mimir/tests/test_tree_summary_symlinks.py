"""A dangling symlink costs the tree summary one entry, not the whole call."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"

from mimir.servers.workspace import server_search as ss


class DanglingSymlinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self._old_root = ss.SEARCH_ROOT
        ss.SEARCH_ROOT = self.root
        ss._TREE_CACHE.clear()
        venv = os.path.join(self.root, ".venv39", "bin")
        os.makedirs(venv)
        os.symlink("/nowhere/that/exists/bin/python", os.path.join(venv, "python"))
        with open(os.path.join(self.root, "real.py"), "w") as fh:
            fh.write("x = 1\n")

    def tearDown(self) -> None:
        ss.SEARCH_ROOT = self._old_root
        ss._TREE_CACHE.clear()
        self._tmp.cleanup()

    def test_the_summary_survives_and_still_reports_the_real_files(self) -> None:
        result = ss.tree_summary(self.root, max_depth=4, use_cache=False)
        self.assertEqual(result.get("status"), "ok", result)
        self.assertIn("real.py", result["tree"])

    def test_the_broken_link_is_listed_and_named_as_one(self) -> None:
        tree = ss.tree_summary(self.root, max_depth=4, use_cache=False)["tree"]
        self.assertIn("python [broken link]", tree)


if __name__ == "__main__":
    unittest.main()

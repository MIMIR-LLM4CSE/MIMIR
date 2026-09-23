"""A dangling symlink must cost the tree summary one entry, not the whole call.

Incident (2026-09-22): the first tool call of a session, ``tree_summary(".")``, died
with ``[Errno 2] No such file or directory: '…/.venv39/bin/python'``. The path was
not an interpreter MIMIR went looking for — it was an entry in the tree being
summarised: a virtualenv whose base interpreter had moved, leaving ``bin/python`` a
symlink to nothing. ``os.walk`` lists such a file, ``os.path.getsize`` raises on it,
and the exception left the model with no map of the workspace at all.

``list_files`` never had the bug — it sizes with ``if os.path.isfile(fp) else None``,
which reads False on a dangling link — which is why the same session's
``list_directory`` succeeded in the same millisecond.

Pure-Python + temp dirs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "workspace"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_search as ss  # noqa: E402


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

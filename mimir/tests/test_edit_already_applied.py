"""replace_in_file reports a clear no-op when the edit was already applied.

Paths are absolute throughout: file tools reject relative paths (see
``server_files._require_abs``), so ``_write`` returns the absolute path to use.

Root cause of "I see an edit failure but the edit is actually done": the model
re-issues an edit whose anchor a prior successful call already replaced. The
anchor is gone, so the naive result was "Target text was not found" (reads as a
failure) even though the file already holds the intended change. Now that case
returns a benign no-op (✓), while a genuine wrong anchor still errors.

Run:
    python -m unittest mimir.tests.test_edit_already_applied -v
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared", _SERVERS / "workspace"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import server_files as sf  # noqa: E402


class AlreadyAppliedTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_root = sf.ROOT_DIR_ABS
        sf.ROOT_DIR_ABS = self._tmp.name

    def tearDown(self) -> None:
        sf.ROOT_DIR_ABS = self._old_root
        self._tmp.cleanup()

    def _write(self, text: str) -> str:
        p = os.path.join(self._tmp.name, "t.py")
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def test_reapplied_replacement_is_noop_not_error(self) -> None:
        fp = self._write("beta = 2\nkeep = 0\n")   # already holds the intended result
        res = sf.replace_in_file(path=fp, old_text="alpha = 1", new_text="beta = 2")
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res.get("operation"), "noop")
        self.assertIn("already applied", res.get("reason", ""))

    def test_reapplied_deletion_is_noop(self) -> None:
        fp = self._write("keep = 0\n")             # the line to delete is already gone
        res = sf.replace_in_file(path=fp, old_text="gamma = 3\n", new_text="")
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res.get("operation"), "noop")

    def test_genuine_wrong_anchor_still_errors(self) -> None:
        # new_text is NOT present → this is a real miss, not an already-applied edit.
        fp = self._write("beta = 2\n")
        res = sf.replace_in_file(path=fp, old_text="alpha = 1", new_text="zeta = 9")
        self.assertEqual(res.get("status"), "error")
        self.assertIn("not found", res.get("error", "").lower())

    def test_normal_edit_unaffected(self) -> None:
        fp = self._write("N = 10\n")
        res = sf.replace_in_file(path=fp, old_text="N = 10", new_text="N = 20")
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res.get("operation"), "replaced")

    def test_replace_all_already_applied_is_noop(self) -> None:
        fp = self._write("beta = 2\nbeta = 2\n")
        res = sf.replace_all_in_file(path=fp, old_text="alpha = 1",
                                     new_text="beta = 2", confirm=True)
        self.assertEqual(res.get("status"), "ok")
        self.assertEqual(res.get("operation"), "noop")

    def test_replace_all_genuine_miss_still_errors(self) -> None:
        fp = self._write("beta = 2\n")
        res = sf.replace_all_in_file(path=fp, old_text="alpha = 1",
                                     new_text="zeta9 = 9", confirm=True)
        self.assertEqual(res.get("status"), "error")

    def test_apply_edits_skips_already_applied_subedit(self) -> None:
        import json
        fp = self._write("beta = 2\nN = 10\n")
        edits = json.dumps([
            {"operation": "replace_in_file", "path": fp,
             "old_text": "alpha = 1", "new_text": "beta = 2"},   # already applied
            {"operation": "replace_in_file", "path": fp,
             "old_text": "N = 10", "new_text": "N = 20"},        # real
        ])
        res = sf.apply_edits(edits_json=edits, confirm=True)
        self.assertEqual(res.get("status"), "ok")
        statuses = [e.get("status") for e in res.get("per_edit_results", [])]
        self.assertIn("noop", statuses)
        with open(os.path.join(self._tmp.name, "t.py")) as fh:
            self.assertEqual(fh.read(), "beta = 2\nN = 20\n")


if __name__ == "__main__":
    unittest.main()

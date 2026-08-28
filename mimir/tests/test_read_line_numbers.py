"""A read states the number of every line it returns.

Root cause this covers: the edit tools are addressed by line number, and the read that
was supposed to supply those numbers handed back an unnumbered block with only the
window's own bounds around it. Mapping "this line of text" to "line 203" was then an
offset the model had to count by hand — and its way out of the uncertainty was to keep
re-reading narrower ranges (195-215, 200-210, 200-225, 203, 203) until the requested
range was a single line and the number was asserted by the arguments instead of
inferred. Numbering the content removes the thing those reads were resolving.

The second half is the cost of that: the text a model now holds is numbered, and the
anchors and replacement bodies it is asked for are not.

Run:
    python -m unittest mimir.tests.test_read_line_numbers -v
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
import server_search as ss  # noqa: E402


class ReadNumbersItsLinesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_root = ss.SEARCH_ROOT
        ss.SEARCH_ROOT = self._tmp.name
        self.addCleanup(lambda: setattr(ss, "SEARCH_ROOT", self._old_root))
        self.path = os.path.join(self._tmp.name, "solver.f90")
        with open(self.path, "w") as fh:
            fh.write("".join(f"  call step_{i}()\n" for i in range(1, 301)))

    def _read(self, start: int, end: int) -> dict:
        return ss.read_file_lines(self.path, start, end)

    def test_every_returned_line_carries_its_number_in_the_file(self) -> None:
        payload = self._read(200, 210)
        lines = payload["content"].splitlines()
        self.assertEqual(lines[0], "200:   call step_200()")
        self.assertEqual(lines[-1], "210:   call step_210()")

    def test_the_number_is_the_file_s_own_not_the_window_s_offset(self) -> None:
        # The whole point: a window opened at 200 must not call its first line 1.
        self.assertIn("203:   call step_203()", self._read(195, 215)["content"])

    def test_indentation_survives_the_prefix(self) -> None:
        # An anchor is only usable if the whitespace after the prefix is the file's.
        line = self._read(7, 7)["content"]
        self.assertEqual(line, "7:   call step_7()")

    def test_an_empty_window_returns_nothing_rather_than_a_bare_number(self) -> None:
        payload = self._read(400, 410)
        self.assertEqual(payload["content"], "")
        self.assertEqual(payload["lines_returned"], 0)


class NumberedTextComingBackAsAnEditTests(unittest.TestCase):
    """What the model holds is numbered; what the edit tools take is not."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._old_root = sf.ROOT_DIR_ABS
        sf.ROOT_DIR_ABS = self._tmp.name
        self.addCleanup(lambda: setattr(sf, "ROOT_DIR_ABS", self._old_root))

    def _write(self, text: str, name: str = "t.py") -> str:
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def _body(self, path: str) -> str:
        with open(path) as fh:
            return fh.read()

    def test_an_anchor_copied_with_its_prefixes_still_finds_its_target(self) -> None:
        fp = self._write("a = 1\nb = 2\nc = 3\n")
        res = sf.replace_in_file(
            path=fp, old_text="2: b = 2\n3: c = 3", new_text="b = 9\nc = 9")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self._body(fp), "a = 1\nb = 9\nc = 9\n")

    def test_a_replacement_body_pasted_back_numbered_is_written_unnumbered(self) -> None:
        fp = self._write("a = 1\nb = 2\nc = 3\n")
        res = sf.replace_lines(
            path=fp, start_line=2, end_line=3, new_content="2: b = 9\n3: c = 9\n")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(self._body(fp), "a = 1\nb = 9\nc = 9\n")

    def test_a_body_numbered_from_somewhere_else_is_written_as_given(self) -> None:
        # Numbers that do not start at the range being replaced are not a paste-back of
        # it, and guessing otherwise would silently delete text the caller meant to keep.
        fp = self._write("a = 1\nb = 2\n", name="t.txt")
        sf.replace_lines(path=fp, start_line=2, end_line=2, new_content="90: b = 9\n")
        self.assertEqual(self._body(fp), "a = 1\n90: b = 9\n")

    def test_code_that_merely_looks_numbered_is_matched_literally(self) -> None:
        # A mapping entry is not a numbered read, and stripping it would rewrite the
        # wrong line. The literal anchor matches first, so the question never arises.
        fp = self._write("cfg = {\n  1: 'alpha',\n  2: 'beta',\n}\n", name="t.txt")
        res = sf.replace_in_file(path=fp, old_text="  1: 'alpha',", new_text="  1: 'ALPHA',")
        self.assertEqual(res["status"], "ok")
        self.assertIn("1: 'ALPHA',", self._body(fp))

    def test_a_non_consecutive_prefix_run_is_not_treated_as_numbering(self) -> None:
        self.assertIsNone(sf._strip_line_number_prefixes("10: a\n40: b"))

    def test_unnumbered_text_is_left_alone(self) -> None:
        self.assertIsNone(sf._strip_line_number_prefixes("a = 1\nb = 2"))


if __name__ == "__main__":
    unittest.main()

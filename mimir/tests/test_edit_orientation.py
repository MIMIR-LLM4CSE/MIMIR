"""A successful edit tells the caller where it landed in the file it just produced.

Root cause this covers: an edit renumbers every line below it, so the line numbers the
model used to find the site — from a search, an outline, an earlier read — stop
describing the file the instant the write succeeds. When the result carried only a diff,
the only way back to a usable position was another search plus another read, per edit.
Now the result carries the new span, the shift applied below it, and the region as it
now stands.

Run:
    python -m unittest mimir.tests.test_edit_orientation -v
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


class ChangedSpanTests(unittest.TestCase):
    """The span is derived from the two contents, not from any tool's arguments."""

    def _span(self, old: str, new: str) -> tuple[int, int]:
        return sf._changed_line_span(old.splitlines(True), new.splitlines(True))

    def test_single_line_replacement(self) -> None:
        self.assertEqual(self._span("a\nb\nc\n", "a\nB\nc\n"), (2, 2))

    def test_insertion_reports_the_new_lines(self) -> None:
        self.assertEqual(self._span("a\nc\n", "a\nb1\nb2\nc\n"), (2, 3))

    def test_deletion_reports_an_empty_span_at_the_site(self) -> None:
        start, end = self._span("a\nb\nc\n", "a\nc\n")
        self.assertEqual(start, 2)
        self.assertLess(end, start)

    def test_append_at_end_of_file(self) -> None:
        self.assertEqual(self._span("a\n", "a\nb\n"), (2, 2))


class OrientationPayloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._old_root = sf.ROOT_DIR_ABS
        sf.ROOT_DIR_ABS = self._tmp.name

    def tearDown(self) -> None:
        sf.ROOT_DIR_ABS = self._old_root
        self._tmp.cleanup()

    def _write(self, text: str, name: str = "t.py") -> str:
        p = os.path.join(self._tmp.name, name)
        with open(p, "w") as fh:
            fh.write(text)
        return p

    def _numbered_body(self, n: int) -> str:
        return "".join(f"line_{i} = {i}\n" for i in range(1, n + 1))

    def test_replace_in_file_reports_new_position_and_shift(self) -> None:
        fp = self._write(self._numbered_body(60))
        res = sf.replace_in_file(
            path=fp, old_text="line_30 = 30", new_text="line_30 = 30\nline_30b = 300")

        self.assertEqual(res["status"], "ok")
        # The anchor line is reproduced verbatim, so the minimal changed region is the
        # inserted line alone — the span reports what actually moved, not what was sent.
        self.assertEqual(res["new_start_line"], 31)
        self.assertEqual(res["new_end_line"], 31)
        self.assertEqual(res["total_lines"], 61)
        self.assertEqual(res["line_delta"], 1)

    def test_the_span_describes_the_file_as_it_now_stands(self) -> None:
        fp = self._write(self._numbered_body(60))
        sf.replace_in_file(
            path=fp, old_text="line_10 = 10", new_text="line_10 = 10\nINSERTED = 1")
        res = sf.replace_in_file(path=fp, old_text="line_40 = 40", new_text="line_40 = 41")

        # line_40 sat at 40 before the first insert and at 41 after it; the span must
        # describe the file as it is now, or it is worse than useless.
        self.assertEqual(res["new_start_line"], 41)
        self.assertEqual(res["new_end_line"], 41)
        self.assertEqual(res["total_lines"], 61)

    def test_the_changed_region_is_left_to_the_diff(self) -> None:
        # The unified diff already quotes the region; repeating it as a numbered window
        # put the same text in the payload twice.
        fp = self._write("a = 1\nb = 2\n")
        res = sf.replace_in_file(path=fp, old_text="a = 1", new_text="a = 9")
        self.assertNotIn("context", res)
        self.assertIn("a = 9", res["diff"])

    def test_replace_lines_carries_orientation(self) -> None:
        fp = self._write(self._numbered_body(40))
        res = sf.replace_lines(
            path=fp, start_line=5, end_line=6, new_content="merged = 1\n")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["new_start_line"], 5)
        self.assertEqual(res["line_delta"], -1)
        self.assertEqual(res["total_lines"], 39)

    def test_replace_all_carries_orientation_only_once_applied(self) -> None:
        fp = self._write("v = alpha\nw = alpha\n", name="t.txt")
        preview = sf.replace_all_in_file(
            path=fp, old_text="alpha", new_text="beta", confirm=False)
        self.assertNotIn("new_start_line", preview)

        applied = sf.replace_all_in_file(
            path=fp, old_text="alpha", new_text="beta", confirm=True)
        self.assertEqual(applied["new_start_line"], 1)
        self.assertEqual(applied["line_delta"], 0)


class AnchorMissHandsBackTheAnchorTests(unittest.TestCase):
    """A failed anchor is a request for the current text — answer it with the text.

    Sending the caller off to read the file is only right when nothing in the reply
    shows the site; when the excerpt is right there, ordering a read anyway is what
    turns one failed edit into a search plus a read.
    """

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

    def test_near_match_points_at_the_excerpt_not_at_a_read(self) -> None:
        fp = self._write("a = 0\nb = 0\nvalue    =   compute()\nc = 0\nd = 0\n")
        res = sf.replace_in_file(
            path=fp, old_text="value = compute()", new_text="value = compute_v2()")

        self.assertEqual(res["status"], "error")
        self.assertIn("Closest matches", res["anchor_excerpt"])
        self.assertIn("Copy old_text verbatim from anchor_excerpt", res["hint"])
        self.assertNotIn("Use read_file_lines", res["hint"])

    def test_excerpt_preserves_the_exact_spacing_that_failed(self) -> None:
        # err() collapses whitespace in the message, so an excerpt carried there would
        # be unusable as an anchor — the whole point of handing it back.
        fp = self._write("a = 0\nvalue    =   compute()\nb = 0\n")
        res = sf.replace_in_file(
            path=fp, old_text="value = compute()", new_text="x")
        self.assertIn("2: value    =   compute()", res["anchor_excerpt"])

    def test_excerpt_carries_lines_either_side_of_the_near_match(self) -> None:
        fp = self._write("".join(f"pad_{i} = {i}\n" for i in range(1, 21))
                         + "value    =   compute()\n"
                         + "".join(f"tail_{i} = {i}\n" for i in range(1, 21)))
        res = sf.replace_in_file(
            path=fp, old_text="value = compute()", new_text="x")
        excerpt = res["anchor_excerpt"]
        self.assertIn(f"{21 - sf._ANCHOR_SNIPPET_MARGIN}: pad_", excerpt)
        self.assertIn(f"{21 + sf._ANCHOR_SNIPPET_MARGIN}: tail_", excerpt)

    def test_no_near_match_still_directs_to_a_read(self) -> None:
        fp = self._write("a = 0\nb = 0\n")
        res = sf.replace_in_file(
            path=fp, old_text="nothing_like_this_exists", new_text="x")
        self.assertEqual(res["status"], "error")
        self.assertNotIn("anchor_excerpt", res)
        self.assertIn("Use read_file_lines", res["hint"])


if __name__ == "__main__":
    unittest.main()

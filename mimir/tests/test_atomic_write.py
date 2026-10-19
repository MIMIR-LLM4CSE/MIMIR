"""shared/atomic: a reader sees the old file or the new one, never half."""

import json
import os
import tempfile
import unittest

from mimir.shared.atomic import replace_symlink, write_json, write_text


class WriteJsonTests(unittest.TestCase):
    def test_writes_the_whole_file_and_leaves_nothing_beside_it(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "state.json")
            write_json(path, {"a": 1}, indent=2, trailing_newline=True)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertEqual(json.loads(text), {"a": 1})
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(os.listdir(os.path.dirname(path)), ["state.json"])

    def test_a_failed_write_keeps_the_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            write_json(path, {"v": 1})
            with self.assertRaises(TypeError):
                write_json(path, {"v": object()})
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), {"v": 1})
            self.assertEqual(os.listdir(d), ["state.json"])


class WriteTextTests(unittest.TestCase):
    def test_newline_is_passed_through(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            write_text(path, "a\r\nb\n", newline="")
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), b"a\r\nb\n")

    def test_replaces_an_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "f.txt")
            write_text(path, "old")
            write_text(path, "new")
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), "new")
            self.assertEqual(os.listdir(d), ["f.txt"])


class ReplaceSymlinkTests(unittest.TestCase):
    def test_repoints_an_existing_link(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a"), os.path.join(d, "b")
            os.mkdir(a)
            os.mkdir(b)
            link = os.path.join(d, "active")
            replace_symlink(link, a)
            replace_symlink(link, b)
            self.assertEqual(os.readlink(link), b)
            self.assertEqual(sorted(os.listdir(d)), ["a", "active", "b"])


if __name__ == "__main__":
    unittest.main()

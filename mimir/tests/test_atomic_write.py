"""The one way the client writes a JSON file that something else reads."""

import json
import os
import tempfile
import unittest

from mimir.client.util.files import write_json_atomic


class WriteJsonAtomicTests(unittest.TestCase):
    def test_writes_the_whole_file_and_leaves_nothing_beside_it(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "state.json")
            write_json_atomic(path, {"a": 1}, indent=2, trailing_newline=True)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            self.assertEqual(json.loads(text), {"a": 1})
            self.assertTrue(text.endswith("\n"))
            self.assertEqual(os.listdir(os.path.dirname(path)), ["state.json"])

    def test_a_failed_write_keeps_the_previous_file(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            write_json_atomic(path, {"v": 1})
            with self.assertRaises(TypeError):
                write_json_atomic(path, {"v": object()})
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh), {"v": 1})
            self.assertEqual(os.listdir(d), ["state.json"])


if __name__ == "__main__":
    unittest.main()

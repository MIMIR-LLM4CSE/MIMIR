"""Hermetic tests for the file target of a tool row (tool_execution/file_target.py).

The target is what makes the file name in a row clickable: the absolute path the
call touched and the lines it covered. Found by argument name and result keys, never
by tool name, and only for an existing file.
"""

import json
import os
import tempfile
import unittest

from mimir.client.tool_execution.file_target import file_target, result_span


class FileTargetTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".py")
        os.close(fd)

    def tearDown(self):
        os.unlink(self.path)

    def test_read_carries_the_range_it_returned(self):
        result = json.dumps({"status": "ok", "start_line": 50, "end_line": 80})
        target = file_target("any_tool", {"path": self.path}, result)
        self.assertEqual(target, {
            "path": self.path, "name": os.path.basename(self.path),
            "line": 50, "end_line": 80,
        })

    def test_edit_span_wins_over_the_read_keys(self):
        result = json.dumps({"new_start_line": 12, "new_end_line": 14,
                             "start_line": 1, "end_line": 2})
        target = file_target("any_tool", {"filepath": self.path}, result)
        self.assertEqual((target["line"], target["end_line"]), (12, 14))

    def test_pure_deletion_keeps_the_line_alone(self):
        self.assertEqual(result_span(json.dumps({"new_start_line": 7, "new_end_line": 6})),
                         {"line": 7})

    def test_result_without_lines_opens_the_file_top(self):
        target = file_target("any_tool", {"path": self.path}, "plain text")
        self.assertNotIn("line", target)

    def test_relative_path_gives_no_target(self):
        self.assertIsNone(file_target("any_tool", {"path": "foo.py"}, "{}"))

    def test_missing_file_gives_no_target(self):
        self.assertIsNone(file_target("any_tool", {"path": self.path + ".gone"}, "{}"))

    def test_directory_gives_no_target(self):
        self.assertIsNone(file_target("any_tool", {"path": os.path.dirname(self.path)}, "{}"))

    def test_bogus_line_values_are_ignored(self):
        self.assertEqual(result_span(json.dumps({"start_line": True, "end_line": 3})), {})
        self.assertEqual(result_span(json.dumps({"start_line": 0})), {})


if __name__ == "__main__":
    unittest.main()

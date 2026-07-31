"""Hermetic tests for the registry-driven pre-write diff previews.

``ui/file_preview.py`` reconstructs a proposed post-write diff from a tool call's
arguments, dispatching on the generic ``preview`` kind the tool declares — no
tool names in shipped code. These tests build small registries via the real
server-side descriptor helper so declaration and consumption are exercised
end-to-end, against real files in a temp workspace.
"""

import os
import tempfile
import types
import unittest

from mimir.client.context import capabilities as caps
from mimir.client.ui.ws import file_preview
from mimir.servers._shared import capabilities as srv
from mimir.tests import _golden_caps as golden


def _tool(name: str, **descriptor_kwargs) -> dict:
    """{name: ToolCaps} registry entry declared through the real helper."""
    desc = srv.build_descriptor(**descriptor_kwargs)
    fake = types.SimpleNamespace(name=name, meta={"mimir": desc},
                                 annotations=None, inputSchema=None)
    return {name: caps.infer_tool_caps(fake)}


class PreviewKindCoverage(unittest.TestCase):
    def test_every_declared_kind_has_a_builder(self):
        """A server can't declare a preview shape the client can't render."""
        declared = set(golden.PREVIEW_KIND_BY_TOOL.values())
        self.assertTrue(declared <= set(file_preview._PREVIEW_KINDS),
                        f"kinds without builders: {declared - set(file_preview._PREVIEW_KINDS)}")


class BuildPreviewDiffs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def _write(self, rel: str, content: str) -> str:
        path = os.path.join(self.root, rel)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def _diffs(self, reg, tool, arguments):
        return file_preview.build_preview_diffs(tool, arguments, reg, root=self.root)

    # ── shapes ──────────────────────────────────────────────────────────────

    def test_content_write_existing_file(self):
        self._write("a.py", "old\n")
        reg = _tool("w", preview={"kind": "content", "args": ["content"]})
        entries = self._diffs(reg, "w", {"path": "a.py", "content": "new\n"})
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["file"], "a.py")
        self.assertIn("-old", entries[0]["patch"])
        self.assertIn("+new", entries[0]["patch"])
        # Whole-content writes carry the proposed content for a real diff editor.
        self.assertEqual(entries[0]["new_content"], "new\n")
        self.assertNotIn("is_new", entries[0])

    def test_content_write_new_file_flagged(self):
        reg = _tool("w", preview={"kind": "content", "args": ["content"]})
        entries = self._diffs(reg, "w", {"path": "fresh.py", "content": "x = 1\n"})
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["is_new"])
        self.assertIn("+x = 1", entries[0]["patch"])

    def test_append(self):
        self._write("log.txt", "one\n")
        reg = _tool("a", preview={"kind": "append", "args": ["content"]})
        entries = self._diffs(reg, "a", {"path": "log.txt", "content": "two\n"})
        self.assertEqual(len(entries), 1)
        self.assertIn("+two", entries[0]["patch"])
        self.assertNotIn("-one", entries[0]["patch"])
        self.assertNotIn("new_content", entries[0])  # only whole-content writes

    def test_replace_single_occurrence(self):
        self._write("b.py", "x = 1\ny = x\n")
        reg = _tool("r", preview={"kind": "replace", "args": ["old_text", "new_text"]})
        entries = self._diffs(reg, "r", {"path": "b.py", "old_text": "x", "new_text": "z"})
        self.assertEqual(len(entries), 1)
        # Only the first occurrence changes.
        self.assertIn("+z = 1", entries[0]["patch"])
        self.assertNotIn("+y = z", entries[0]["patch"])

    def test_replace_all_occurrences(self):
        self._write("c.py", "x = 1\ny = x\n")
        reg = _tool("ra", preview={"kind": "replace_all", "args": ["old_text", "new_text"]})
        entries = self._diffs(reg, "ra", {"path": "c.py", "old_text": "x", "new_text": "z"})
        self.assertIn("+z = 1", entries[0]["patch"])
        self.assertIn("+y = z", entries[0]["patch"])

    def test_replace_empty_old_text_yields_nothing(self):
        self._write("d.py", "content\n")
        reg = _tool("r", preview={"kind": "replace", "args": ["old_text", "new_text"]})
        self.assertEqual(self._diffs(reg, "r", {"path": "d.py", "old_text": "", "new_text": "z"}), [])

    def test_line_splice(self):
        self._write("e.py", "l1\nl2\nl3\n")
        reg = _tool("rl", preview={"kind": "line_splice",
                                   "args": ["start_line", "end_line", "new_content"]})
        entries = self._diffs(reg, "rl", {"path": "e.py", "start_line": 2,
                                          "end_line": 2, "new_content": "L2"})
        self.assertIn("-l2", entries[0]["patch"])
        self.assertIn("+L2", entries[0]["patch"])
        self.assertNotIn("-l3", entries[0]["patch"])

    def test_delete_renders_removal_and_flags(self):
        self._write("gone.py", "a\nb\n")
        reg = _tool("d", preview={"kind": "delete"})
        entries = self._diffs(reg, "d", {"path": "gone.py"})
        self.assertEqual(len(entries), 1)
        self.assertTrue(entries[0]["is_delete"])
        self.assertIn("-a", entries[0]["patch"])
        self.assertIn("-b", entries[0]["patch"])

    def test_delete_missing_file_yields_nothing(self):
        reg = _tool("d", preview={"kind": "delete"})
        self.assertEqual(self._diffs(reg, "d", {"path": "absent.py"}), [])

    # ── gating ──────────────────────────────────────────────────────────────

    def test_no_preview_spec_yields_nothing(self):
        """A tool without a preview declaration (e.g. a foreign-server writer or a
        non-file mutation like an env delete) is not previewable — the caller must
        fall through to the normal approval prompt."""
        self._write("f.py", "old\n")
        reg = _tool("foreign_edit_file")  # no descriptor fields at all
        self.assertEqual(self._diffs(reg, "foreign_edit_file",
                                     {"path": "f.py", "content": "new\n"}), [])
        self.assertIsNone(caps.preview_spec("foreign_edit_file", reg))

    def test_unknown_kind_yields_nothing(self):
        self._write("g.py", "old\n")
        c = caps.ToolCaps(name="odd", preview={"kind": "hologram"})
        self.assertEqual(self._diffs({"odd": c}, "odd", {"path": "g.py"}), [])

    def test_missing_path_yields_nothing(self):
        reg = _tool("w", preview={"kind": "content", "args": ["content"]})
        self.assertEqual(self._diffs(reg, "w", {"content": "new\n"}), [])

    def test_no_change_yields_nothing(self):
        self._write("same.py", "same\n")
        reg = _tool("w", preview={"kind": "content", "args": ["content"]})
        self.assertEqual(self._diffs(reg, "w", {"path": "same.py", "content": "same\n"}), [])

    def test_declared_path_role_wins(self):
        """When the tool declares its path arg-role, it is used instead of the
        generic path/filepath fallback names."""
        self._write("h.py", "old\n")
        reg = _tool("w", path_args=["target"],
                    preview={"kind": "content", "args": ["content"]})
        entries = self._diffs(reg, "w", {"target": "h.py", "content": "new\n"})
        self.assertEqual(entries[0]["file"], "h.py")


if __name__ == "__main__":
    unittest.main()

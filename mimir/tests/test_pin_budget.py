"""Tests for the checklist section of the system prompt and the recency sets.

The checklist is state the model is held to, so what the prompt says about it is
load-bearing. It carries the checklist and nothing else: it used to also repeat the
paths read, written and planned this session, which the model copied instead of used
(one recorded run looped on the file list until the step budget ran out). It lives in
messages[0] — never in the last position before the generation prompt, which is what
emptied turns on a served model (see test_prefix_cache).
"""

import os
import tempfile
import unittest

from mimir.client.context.execution_context import (
    RecencySet,
    execution_context_template,
    recent_first,
    validate_execution_context,
)
from mimir.client.prompt.system_prompt import build_system_content


class RecencySetTests(unittest.TestCase):
    def test_it_is_a_set(self) -> None:
        s = RecencySet(["b", "a"])
        self.assertIsInstance(s, set)
        self.assertIn("a", s)
        self.assertEqual(len(s), 2)

    def test_insertion_order_is_kept(self) -> None:
        s = RecencySet()
        for x in ["z", "a", "m"]:
            s.add(x)
        self.assertEqual(s.insertion_order(), ["z", "a", "m"])

    def test_re_adding_does_not_move_an_entry(self) -> None:
        s = RecencySet(["a", "b"])
        s.add("a")
        self.assertEqual(s.insertion_order(), ["a", "b"])

    def test_discard_removes_from_the_order(self) -> None:
        s = RecencySet(["a", "b", "c"])
        s.discard("b")
        s.discard("absent")  # must not raise
        self.assertEqual(s.insertion_order(), ["a", "c"])

    def test_in_place_difference_keeps_the_order_coherent(self) -> None:
        # _update_carry_context does `carry_reads -= dirty`.
        s = RecencySet(["a", "b", "c"])
        s -= {"b"}
        self.assertIsInstance(s, RecencySet)
        self.assertEqual(s.insertion_order(), ["a", "c"])

    def test_set_operators_degrade_to_a_plain_set(self) -> None:
        # `prior | current` in _update_carry_context: order is lost, nothing breaks.
        merged = RecencySet(["a"]) | {"b"}
        self.assertIsInstance(merged, set)
        self.assertEqual(merged, {"a", "b"})

    def test_recent_first_falls_back_to_sorted_for_a_plain_set(self) -> None:
        self.assertEqual(recent_first({"c", "a", "b"}), ["a", "b", "c"])
        self.assertEqual(recent_first(None), [])

    def test_recent_first_reverses_insertion_order(self) -> None:
        self.assertEqual(recent_first(RecencySet(["a", "b", "c"])), ["c", "b", "a"])

    def test_template_still_validates(self) -> None:
        ctx = execution_context_template()
        ctx["read_files"].add("a.py")
        validate_execution_context(ctx)  # must not raise


class ChecklistSectionTests(unittest.TestCase):
    def _todo(self, body: str) -> str:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "todo_list.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        self.addCleanup(os.rmdir, tmp)
        self.addCleanup(os.remove, path)
        return path

    def _prompt(self, body: str) -> str:
        return build_system_content(
            active_mode="agent", tool_owner={}, sensitive_tools=set(),
            todo_file=self._todo(body),
        )

    def test_it_renders_the_checklist_with_the_pending_count(self) -> None:
        out = self._prompt("- [x] read the solver\n- [ ] add the binding\n")
        self.assertIn("Task checklist (1 pending", out)
        self.assertIn("[x] read the solver", out)
        self.assertIn("[ ] add the binding", out)

    def test_discovery_evidence_is_not_rendered(self) -> None:
        # The paths are already in the transcript; repeating a bare list of them in the
        # prompt is a pattern the model copies rather than uses.
        out = self._prompt("- [ ] add the binding\n")
        for absent in ("Files read", "Known existing paths",
                       "Planned edit targets", "Files written"):
            self.assertNotIn(absent, out)

    def test_no_checklist_says_so_rather_than_rendering_one(self) -> None:
        out = self._prompt("no items\n")
        self.assertIn("No task checklist yet", out)
        self.assertNotIn("Task checklist (", out)

    def test_rendering_is_stable_across_calls(self) -> None:
        # Byte-stability is what lets _sync_checklist skip the rewrite, and with it the
        # prefix-cache break, when the file was touched but not changed.
        path = self._todo("- [ ] a\n- [ ] b\n")
        kw = dict(active_mode="agent", tool_owner={}, sensitive_tools=set(), todo_file=path)
        self.assertEqual(build_system_content(**kw), build_system_content(**kw))


if __name__ == "__main__":
    unittest.main()

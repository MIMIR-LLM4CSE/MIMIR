"""Tests for the per-step pin and the recency-preserving sets (no network).

The pin is re-sent on every step, so what it carries is load-bearing. It carries the
live task checklist and nothing else: it used to also repeat the paths read, written
and planned this session, which the model copied instead of used (a DeepSeek run
looped on the file list until the step budget ran out).
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
from mimir.client.prompt.system_prompt import build_checklist_pin_block


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


class ChecklistPinTests(unittest.TestCase):
    def _ctx_with_todo(self, body: str) -> dict:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "todo_list.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(body)
        self.addCleanup(os.rmdir, tmp)
        self.addCleanup(os.remove, path)
        ctx = execution_context_template()
        ctx["todo_file_path"] = path
        return ctx

    def test_it_renders_the_checklist_with_the_pending_count(self) -> None:
        pin = build_checklist_pin_block(
            self._ctx_with_todo("- [x] read the solver\n- [ ] add the binding\n")
        )
        self.assertIn("Task checklist (1 pending):", pin)
        self.assertIn("  [x] read the solver", pin)
        self.assertIn("  [ ] add the binding", pin)

    def test_discovery_evidence_is_not_pinned(self) -> None:
        # The paths are already in the transcript; repeating them at the tail of every
        # prompt is a pattern the model copies rather than uses.
        ctx = self._ctx_with_todo("- [ ] add the binding\n")
        for i in range(20):
            ctx["read_files"].add(f"src/module_{i:03d}.py")
            ctx["existing_paths"].add(f"src/e_{i:03d}.py")
            ctx["dirty_written_files"].add(f"src/w_{i:03d}.py")
            ctx["planned_edit_targets"].add(f"src/p_{i:03d}.py")
        pin = build_checklist_pin_block(ctx)
        self.assertNotIn("module_", pin)
        self.assertNotIn("Files read", pin)
        self.assertNotIn("Known existing paths", pin)
        self.assertNotIn("Planned edit targets", pin)
        self.assertNotIn("Files written", pin)

    def test_no_checklist_produces_no_pin(self) -> None:
        self.assertEqual(build_checklist_pin_block(execution_context_template()), "")
        self.assertEqual(build_checklist_pin_block(self._ctx_with_todo("no items\n")), "")

    def test_rendering_is_stable_across_calls(self) -> None:
        ctx = self._ctx_with_todo("- [ ] a\n- [ ] b\n")
        first = build_checklist_pin_block(ctx)
        self.assertEqual(build_checklist_pin_block(ctx), first)


if __name__ == "__main__":
    unittest.main()

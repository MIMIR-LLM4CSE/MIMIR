"""Tests for the discovery pin's size bound and its recency ranking (no network).

The pin is re-sent on every step, so what it costs and which slice it shows are both
load-bearing. Two properties are pinned here: every section is capped (three of them
used to print in full, and they were the ones that grow on a long refactor), and the
slice is the most recently touched entries rather than the alphabetically last.
"""

import unittest

from mimir.client.context.execution_context import (
    RecencySet,
    execution_context_template,
    recent_first,
    validate_execution_context,
)
from mimir.client.prompt.system_prompt import build_discovery_pin_block


def _pin_paths(pin: str) -> list[str]:
    """Basenames of the path lines of *pin*, in the order they are rendered."""
    return [
        line.strip().rsplit("/", 1)[-1]
        for line in pin.splitlines()
        if line.startswith("  /")
    ]


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


class PinCapTests(unittest.TestCase):
    def _saturated(self):
        ctx = execution_context_template()
        for i in range(200):
            ctx["read_files"].add(f"src/module_{i:03d}.py")
            ctx["existing_paths"].add(f"src/e_{i:03d}.py")
        for i in range(100):
            ctx["search_queries_used"].add(f"pattern_{i:03d}")
        for i in range(40):
            ctx["dirty_written_files"].add(f"src/w_{i:03d}.py")
        for i in range(20):
            ctx["planned_edit_targets"].add(f"src/p_{i:03d}.py")
        ctx["prev_query_written_files"] = {f"src/q_{i:03d}.py" for i in range(20)}
        return ctx

    def test_every_section_is_capped(self) -> None:
        pin = build_discovery_pin_block(self._saturated(), max_files=10, max_queries=5)
        shown = _pin_paths(pin)
        for stem, cap in (("module_", 10), ("e_", 10), ("w_", 10), ("p_", 10), ("q_", 10)):
            # Match the basename's own prefix: a plain substring test would count
            # "module_190.py" as an "e_" entry.
            got = [b for b in shown if b.startswith(stem)]
            self.assertLessEqual(len(got), cap, f"{stem}* is not capped: {len(got)} shown")
        patterns = [ln for ln in pin.splitlines() if ln.strip().startswith("'pattern_")]
        self.assertLessEqual(len(patterns), 5)

    def test_write_side_sections_report_what_they_hide(self) -> None:
        # These three printed in full before; a truncated list must say so.
        pin = build_discovery_pin_block(self._saturated(), max_files=10)
        self.assertIn("... and 30 more", pin)  # 40 written
        self.assertIn("... and 10 more", pin)  # 20 planned / 20 previous-query

    def test_saturated_pin_stays_bounded(self) -> None:
        pin = build_discovery_pin_block(self._saturated(), max_files=10, max_queries=5)
        # 6 sections x (10 entries + header + "more") — comfortably under the ~6.5k
        # chars the uncapped version produced for the same state.
        self.assertLess(len(pin), 4000)

    def test_empty_context_produces_no_pin(self) -> None:
        self.assertEqual(build_discovery_pin_block(execution_context_template()), "")


class PinRecencyTests(unittest.TestCase):
    def test_the_most_recent_read_is_shown(self) -> None:
        ctx = execution_context_template()
        for i in range(12):
            ctx["read_files"].add(f"src/zpad_{i:02d}.py")
        ctx["read_files"].add("src/a_solver.py")  # last read, first alphabetically
        shown = _pin_paths(build_discovery_pin_block(ctx, max_files=10))
        self.assertEqual(shown[0], "a_solver.py")

    def test_ordering_is_recency_not_alphabetical(self) -> None:
        ctx = execution_context_template()
        for name in ["src/c.py", "src/a.py", "src/b.py"]:
            ctx["read_files"].add(name)
        self.assertEqual(_pin_paths(build_discovery_pin_block(ctx)), ["b.py", "a.py", "c.py"])

    def test_rendering_is_stable_across_calls(self) -> None:
        ctx = execution_context_template()
        for i in range(30):
            ctx["read_files"].add(f"src/f_{i:03d}.py")
        first = build_discovery_pin_block(ctx)
        self.assertEqual(build_discovery_pin_block(ctx), first)


if __name__ == "__main__":
    unittest.main()

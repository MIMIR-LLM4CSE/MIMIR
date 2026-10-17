"""A sub-agent gets a share of the window, sized by what it was given.

A child exists to keep its reading out of the caller's context. Left at the model's
full window, several children fanned out on one endpoint each budget for the whole
thing — the fan-out becomes the cost it was meant to avoid, and the trimming,
eviction and compaction inside each child never engage until it has read as much as
the parent could have. So the child carries a ceiling: 64k for an explorer, which
reads a handful of files and returns a conclusion but owns the breadth of its sweep,
and the standing full budget for a working child, which reads what it must to change
code and reports what running it produced.

The ceiling only ever lowers: a model whose window is smaller still sizes the budget.

Pure-Python (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from mimir.client.config.limits import CTX_RESERVED_RATIO, CTX_TOTAL_COMPACT, CTX_TOTAL_FULL
from mimir.client.query_engine.context_budget import context_budget_for
from mimir.servers.agent_state import spawn_limits


class _WindowFixture(unittest.TestCase):
    def budget(self, window, mode="full", ceiling=None):
        backend = type("B", (), {"context_window": lambda self, m: window})()
        with patch("mimir.client.query_engine.context_budget.get_backend",
                   lambda: backend):
            return context_budget_for("m", mode, ceiling=ceiling)


class CeilingLowersTheBudgetTests(_WindowFixture):
    def test_a_ceiling_under_the_window_is_what_the_child_gets(self) -> None:
        total, reserved, _t, _c = self.budget(524_288, ceiling=32_000)
        self.assertEqual(total, 32_000)
        self.assertEqual(reserved, int(32_000 * CTX_RESERVED_RATIO))

    def test_a_smaller_window_still_wins_over_the_ceiling(self) -> None:
        total, _r, _t, _c = self.budget(16_384, ceiling=200_000)
        self.assertEqual(total, 16_384)

    def test_no_ceiling_leaves_the_budget_exactly_as_before(self) -> None:
        self.assertEqual(self.budget(524_288), self.budget(524_288, ceiling=None))

    def test_a_ceiling_above_the_window_changes_nothing(self) -> None:
        self.assertEqual(self.budget(262_144, ceiling=999_999),
                         self.budget(262_144))

    def test_the_derived_budgets_follow_the_ceiling_down(self) -> None:
        total, reserved, trim, compact = self.budget(524_288, ceiling=32_000)
        self.assertEqual(trim, total - reserved)
        self.assertLess(compact, trim)


class UnknownWindowTests(_WindowFixture):
    """With no window to resolve, the ceiling is the only thing that can bind."""

    def test_the_static_default_still_applies_without_a_ceiling(self) -> None:
        total, _r, _t, _c = self.budget(None)
        self.assertEqual(total, CTX_TOTAL_FULL)

    def test_a_ceiling_binds_the_static_default_too(self) -> None:
        total, _r, _t, _c = self.budget(None, ceiling=32_000)
        self.assertEqual(total, 32_000)

    def test_a_ceiling_above_the_static_default_changes_nothing(self) -> None:
        self.assertEqual(self.budget(None, ceiling=999_999), self.budget(None))


class SubagentCeilingsTests(unittest.TestCase):
    def test_an_explorer_gets_room_for_a_wide_sweep(self) -> None:
        explore = spawn_limits.SUBAGENT_CONTEXT_TOKENS_EXPLORE
        self.assertEqual(explore, 64_000)
        # Between the two standing budgets: more than a compact session, so breadth
        # is affordable; well under a working child, whose reading it must not match.
        self.assertGreater(explore, CTX_TOTAL_COMPACT)
        self.assertLess(explore, CTX_TOTAL_FULL)

    def test_a_working_child_gets_the_full_budget(self) -> None:
        self.assertEqual(spawn_limits.SUBAGENT_CONTEXT_TOKENS_WORKING,
                         CTX_TOTAL_FULL)


if __name__ == "__main__":
    unittest.main()

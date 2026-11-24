"""The intervals against values published with the methods, and the exact paired test."""
from __future__ import annotations

import unittest

from mimir.evals.stats import mcnemar, newcombe, wilson


class IntervalTests(unittest.TestCase):
    def test_wilson_matches_its_published_values(self) -> None:
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=4)
        self.assertAlmostEqual(hi, 0.2775, places=4)
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=4)
        self.assertAlmostEqual(hi, 0.7634, places=4)

    def test_nothing_scored_says_nothing(self) -> None:
        self.assertEqual(wilson(0, 0), (0.0, 1.0))

    def test_newcombe_matches_his_worked_example(self) -> None:
        """Newcombe (1998), method 10: 56/70 against 48/80, a gap of 0.2 in (0.0524, 0.3339)."""
        gap, lo, hi = newcombe(48, 80, 56, 70)
        self.assertAlmostEqual(gap, 0.2, places=4)
        self.assertAlmostEqual(lo, 0.0524, places=4)
        self.assertAlmostEqual(hi, 0.3339, places=4)


class McNemarTests(unittest.TestCase):
    def test_six_wins_and_no_loss_is_measurable(self) -> None:
        self.assertAlmostEqual(mcnemar(0, 6), 2 / 64)

    def test_as_many_wins_as_losses_tells_nothing(self) -> None:
        self.assertEqual(mcnemar(5, 5), 1.0)
        self.assertEqual(mcnemar(0, 0), 1.0)

    def test_the_test_is_symmetric(self) -> None:
        self.assertEqual(mcnemar(2, 9), mcnemar(9, 2))


if __name__ == "__main__":
    unittest.main()

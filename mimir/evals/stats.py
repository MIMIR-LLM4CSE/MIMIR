"""Intervals for a pass rate and for the gap between two: what one comparison may conclude.

Wilson's score interval for a proportion, and Newcombe's hybrid score interval (his method
10) for a difference of two independent proportions — both sound at the small counts a
benchmark run gives, where the normal approximation is not.
"""
from __future__ import annotations

import math

Z95 = 1.959963984540054


def wilson(passed: int, n: int, z: float = Z95) -> tuple[float, float]:
    """The interval for *passed* out of *n*; ``(0, 1)`` when nothing was scored."""
    if n == 0:
        return 0.0, 1.0
    p = passed / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(passed_a: int, n_a: int, passed_b: int, n_b: int,
             z: float = Z95) -> tuple[float, float, float]:
    """The gap ``rate_b - rate_a`` and its interval: ``(gap, low, high)``."""
    p_a = passed_a / n_a if n_a else 0.0
    p_b = passed_b / n_b if n_b else 0.0
    lo_a, hi_a = wilson(passed_a, n_a, z)
    lo_b, hi_b = wilson(passed_b, n_b, z)
    gap = p_b - p_a
    low = gap - math.sqrt((p_b - lo_b) ** 2 + (hi_a - p_a) ** 2)
    high = gap + math.sqrt((hi_b - p_b) ** 2 + (p_a - lo_a) ** 2)
    return gap, low, high

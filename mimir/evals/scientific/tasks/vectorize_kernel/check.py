"""Same result to 1e-10 as the original loop, and at least five times faster."""
import importlib.util
import json
import math
import sys
import time

import numpy as np


def _load(workspace):
    spec = importlib.util.spec_from_file_location("kernel", f"{workspace}/kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference(points, charges):
    total = 0.0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = points[i] - points[j]
            total += charges[i] * charges[j] / math.sqrt(float(d @ d))
    return total


def _best_of(fn, *args, repeat=3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*args)
        best = min(best, time.perf_counter() - start)
    return best


def main(workspace):
    rng = np.random.default_rng(7)
    points = rng.uniform(-1, 1, size=(600, 3))
    charges = rng.uniform(-1, 1, size=600)
    try:
        module = _load(workspace)
        got = float(module.potential(points, charges))
    except Exception as exc:
        return False, f"potential does not run: {type(exc).__name__}: {exc}"
    expected = _reference(points, charges)
    if not math.isclose(got, expected, rel_tol=1e-10):
        return False, f"result changed: {got!r} instead of {expected!r}"
    speedup = _best_of(_reference, points, charges) / _best_of(module.potential, points, charges)
    if speedup < 5:
        return False, f"same result, but only {speedup:.1f}x faster (bound 5x)"
    return True, f"same result, {speedup:.0f}x faster"


if __name__ == "__main__":
    passed, detail = main(sys.argv[1])
    print(json.dumps({"passed": passed, "detail": detail}))

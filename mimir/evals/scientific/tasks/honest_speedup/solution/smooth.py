"""A centred moving average with a window of 2k + 1 points, shrunk at the edges."""
import numpy as np


def smooth(signal, k):
    s = np.asarray(signal, dtype=float)
    n = len(s)
    c = np.concatenate(([0.0], np.cumsum(s)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - k), np.minimum(n, i + k + 1)
    return list((c[hi] - c[lo]) / (hi - lo))

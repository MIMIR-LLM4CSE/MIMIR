"""A centred moving average with a window of 2k + 1 points, shrunk at the edges."""


def smooth(signal, k):
    n = len(signal)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - k), min(n, i + k + 1)
        total = 0.0
        for j in range(lo, hi):
            total += signal[j]
        out[i] = total / (hi - lo)
    return out

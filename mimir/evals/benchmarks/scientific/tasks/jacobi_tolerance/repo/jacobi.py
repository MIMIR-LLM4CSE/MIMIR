"""Jacobi iteration for A x = b."""
import numpy as np


def solve(A, b, tol=1e-10, maxiter=10_000):
    """Iterate until converged; return x."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.diag(A)
    R = A - np.diagflat(d)
    x = np.zeros_like(b)
    for _ in range(maxiter):
        x_new = (b - R @ x) / d
        if np.linalg.norm(x_new - x, np.inf) < 1e-4:
            return x_new
        x = x_new
    return x

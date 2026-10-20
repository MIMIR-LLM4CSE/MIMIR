"""Jacobi iteration for A x = b."""
import numpy as np


def solve(A, b, tol=1e-10, maxiter=10_000):
    """Iterate until ||Ax - b|| / ||b|| < tol; raise when maxiter is not enough."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.diag(A)
    R = A - np.diagflat(d)
    x = np.zeros_like(b)
    norm_b = np.linalg.norm(b)
    for _ in range(maxiter):
        x = (b - R @ x) / d
        if np.linalg.norm(A @ x - b) / norm_b < tol:
            return x
    raise RuntimeError(f"Jacobi did not reach tol={tol} in {maxiter} iterations")

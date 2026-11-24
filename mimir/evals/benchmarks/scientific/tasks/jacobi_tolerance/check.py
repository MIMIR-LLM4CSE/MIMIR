"""The relative residual meets each tolerance asked for, and maxiter too small raises."""
import importlib.util
import json
import sys

import numpy as np


def _load(workspace):
    spec = importlib.util.spec_from_file_location("jacobi", f"{workspace}/jacobi.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _system(n=200, seed=3):
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, size=(n, n))
    A += np.diag(np.abs(A).sum(axis=1) * 1.05)   # barely dominant: slow convergence
    return A, rng.uniform(-1, 1, size=n)


def main(workspace):
    try:
        module = _load(workspace)
        A, b = _system()
        for tol in (1e-6, 1e-10, 1e-12):
            x = module.solve(A, b, tol=tol)
            residual = float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))
            if residual >= tol:
                return False, f"asked tol={tol:g}, returned a relative residual of {residual:.1e}"
    except Exception as exc:
        return False, f"solve fails on a convergent system: {type(exc).__name__}: {exc}"
    try:
        module.solve(A, b, tol=1e-12, maxiter=3)
    except Exception:
        return True, "every tolerance met, and a maxiter too small raises"
    return False, "maxiter=3 returned an unconverged answer instead of raising"


if __name__ == "__main__":
    passed, detail = main(sys.argv[1])
    print(json.dumps({"passed": passed, "detail": detail}))

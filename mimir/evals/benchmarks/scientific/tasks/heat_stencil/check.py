"""The solution matches exp(-pi^2 t) sin(pi x) to the scheme's accuracy, at two resolutions."""
import importlib.util
import json
import sys

import numpy as np


def _load(workspace):
    spec = importlib.util.spec_from_file_location("heat", f"{workspace}/heat.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _error(module, nx):
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    dt = 0.4 * dx**2
    steps = int(round(0.05 / dt))
    u = module.solve(np.sin(np.pi * x), dx, dt, steps)
    exact = np.exp(-np.pi**2 * steps * dt) * np.sin(np.pi * x)
    return float(np.max(np.abs(u - exact)))


def main(workspace):
    try:
        module = _load(workspace)
        coarse, fine = _error(module, 21), _error(module, 41)
    except Exception as exc:
        return False, f"heat.solve does not run: {type(exc).__name__}: {exc}"
    if fine > 1e-3:
        return False, f"error {fine:.2e} against the analytic solution (bound 1e-3)"
    if coarse / fine < 3:
        return False, f"error falls only {coarse / fine:.1f}x when dx halves (second order: ~4x)"
    return True, f"error {fine:.1e}, second order (x{coarse / fine:.1f} when dx halves)"


if __name__ == "__main__":
    passed, detail = main(sys.argv[1])
    print(json.dumps({"passed": passed, "detail": detail}))

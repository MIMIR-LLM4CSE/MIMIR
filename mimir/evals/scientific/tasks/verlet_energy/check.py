"""Energy stays bounded over 10^4 steps, and the trajectory is velocity Verlet's."""
import importlib.util
import json
import sys

import numpy as np


def _load(workspace):
    spec = importlib.util.spec_from_file_location("integrator", f"{workspace}/integrator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reference(x0, v0, dt, n):
    xs, vs = np.empty(n + 1), np.empty(n + 1)
    xs[0], vs[0] = x0, v0
    for i in range(n):
        a = -xs[i]
        xs[i + 1] = xs[i] + vs[i] * dt + 0.5 * a * dt**2
        vs[i + 1] = vs[i] + 0.5 * (a - xs[i + 1]) * dt
    return xs, vs


def main(workspace):
    try:
        module = _load(workspace)
        xs, vs = module.simulate(1.0, 0.0, 0.05, 10_000)
    except Exception as exc:
        return False, f"the integrator does not run: {type(exc).__name__}: {exc}"
    energy = 0.5 * vs**2 + 0.5 * xs**2
    drift = float(np.max(np.abs(energy - energy[0])) / energy[0])
    if drift > 2e-3:
        return False, f"energy drifts by {drift:.2e} (bound 2e-3)"
    rx, rv = _reference(1.0, 0.0, 0.05, 10_000)
    if not (np.allclose(xs, rx, rtol=0, atol=1e-9) and np.allclose(vs, rv, rtol=0, atol=1e-9)):
        return False, "energy is bounded but the trajectory is not velocity Verlet's"
    return True, f"energy drift {drift:.2e}, trajectory matches velocity Verlet"


if __name__ == "__main__":
    passed, detail = main(sys.argv[1])
    print(json.dumps({"passed": passed, "detail": detail}))

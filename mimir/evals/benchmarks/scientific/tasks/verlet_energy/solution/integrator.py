"""Velocity-Verlet integration of a particle of unit mass."""
import numpy as np


def harmonic(x):
    """The force of a unit spring."""
    return -x


def step(x, v, dt, force):
    """Advance one time step."""
    a = force(x)
    x_new = x + v * dt + 0.5 * a * dt**2
    v_new = v + 0.5 * (a + force(x_new)) * dt
    return x_new, v_new


def simulate(x0, v0, dt, n, force=harmonic):
    """Positions and velocities over *n* steps, the initial state included."""
    xs = np.empty(n + 1)
    vs = np.empty(n + 1)
    xs[0], vs[0] = x0, v0
    for i in range(n):
        xs[i + 1], vs[i + 1] = step(xs[i], vs[i], dt, force)
    return xs, vs

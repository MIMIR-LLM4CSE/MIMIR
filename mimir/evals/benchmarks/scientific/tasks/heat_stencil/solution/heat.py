"""Explicit finite differences (FTCS) for u_t = u_xx on [0, 1], Dirichlet zero ends."""
import numpy as np


def solve(u0, dx, dt, steps):
    """March *u0* forward *steps* times; ``dt / dx**2`` must stay below 1/2."""
    r = dt / dx**2
    u = np.array(u0, dtype=float)
    for _ in range(steps):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u_new[0] = u_new[-1] = 0.0
        u = u_new
    return u

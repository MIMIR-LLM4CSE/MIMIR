"""The pairwise Coulomb-like potential of a set of point charges."""
import numpy as np


def potential(points, charges):
    """Sum over pairs i < j of q_i q_j / |p_i - p_j|."""
    p = np.asarray(points, dtype=float)
    q = np.asarray(charges, dtype=float)
    i, j = np.triu_indices(len(p), k=1)
    r = np.linalg.norm(p[i] - p[j], axis=1)
    return float(np.sum(q[i] * q[j] / r))

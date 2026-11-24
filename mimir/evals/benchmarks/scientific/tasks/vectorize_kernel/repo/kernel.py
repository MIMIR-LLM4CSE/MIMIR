"""The pairwise Coulomb-like potential of a set of point charges."""
import math


def potential(points, charges):
    """Sum over pairs i < j of q_i q_j / |p_i - p_j|."""
    n = len(points)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dz = points[i][2] - points[j][2]
            total += charges[i] * charges[j] / math.sqrt(dx * dx + dy * dy + dz * dz)
    return total

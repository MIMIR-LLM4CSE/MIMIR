"""Finite-difference derivatives."""


def central_derivative(f, x, h):
    """f'(x) by central differences: error O(h^2)."""
    return (f(x + h) - f(x - h)) / (2 * h)

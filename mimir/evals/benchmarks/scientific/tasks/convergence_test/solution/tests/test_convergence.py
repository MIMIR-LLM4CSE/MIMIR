import math

import numpy as np

from schemes import central_derivative


def test_central_derivative_is_second_order():
    hs = np.array([0.1, 0.05, 0.025, 0.0125])
    errors = [abs(central_derivative(math.sin, 1.0, h) - math.cos(1.0)) for h in hs]
    orders = np.log(np.array(errors[:-1]) / np.array(errors[1:])) / np.log(2.0)
    assert np.all(orders > 1.8), orders

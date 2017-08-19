import pytest
import pyomo.environ as aml
from convexity_detection.bounds import *
from convexity_detection.pyomo_compat import *
import numpy as np


set_pyomo4_expression_tree()


def test_bounds_arithmetic():
    a = Bound(2, 3)
    b = Bound(-6, -3)

    # sanity check
    assert a != b

    # [a, b] + [c, d] = [a + c, b + d]
    assert (a + b) == Bound(-4, 0)
    assert (6.2 + b) == Bound(0.2, 3.2)
    assert (a + 0) == a
    # [a, b] - [c, d] = [a - d, b - c]
    assert (a - b) == Bound(5, 9)
    assert (a - 0) == a
    # [a, b] * [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
    assert (a * b) == Bound(-18, -6)
    assert (a * 0) == Bound(0, 0)
    # [a, b] / [c, d] = [a, b] * [d^-1, c^-1] if a, b !=0 else [-inf, inf]
    assert (a / 1) == a
    assert (a / b) == Bound(-1, -1/3)
    assert (a / 0) == Bound(-np.inf, np.inf)


def test_linear_expr_bounds():
    x = aml.Var(bounds=(-2, 3))
    x.construct()
    y = aml.Var(bounds=(0, None))
    y.construct()
    z = aml.Var(bounds=(1, 2))
    z.construct()
    w = aml.Var(bounds=(4, 10))
    w.construct()

    expr0 = 6*x + 2*y - 10*z
    # [-12, 18] + [0, inf] - [10, 100]
    assert expr_bounds(expr0) == Bound(-32, np.inf)

    expr1 = (x + y) * 2 * z
    # ([-2, 3] + [0, inf]) * [2, 4] = [-2, inf] * [2, 4]
    assert expr_bounds(expr1) == Bound(-8, np.inf)

    expr2 = (x + y) * (z + z*w)
    assert expr_bounds(expr2) == Bound(-44, np.inf)

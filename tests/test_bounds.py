import pytest
import pyomo.environ as aml
from convexity_detection.bounds import *
import numpy as np
from util import _var


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


def test_bound_simple_var():
    assert expr_bounds(_var()) == Bound(None, None)
    assert expr_bounds(_var((None, -3))) == Bound(None, -3)
    assert expr_bounds(_var((-8, 5))) == Bound(-8, 5)


def test_bound_product():
    assert expr_bounds(10*_var()) == Bound(None, None)
    assert expr_bounds(10*_var((-1, 1))) == Bound(-10, 10)
    assert expr_bounds(-10*_var((-2, 1))) == Bound(-10, 20)


def test_bound_linear():
    #    [-12, 18]         + [0, inf]            - [10, 20]
    e0 = 6 * _var((-2, 3)) + 2 * _var((0, None)) - 10 * _var((1, 2))
    assert expr_bounds(e0) == Bound(-32, None)

    #    ([-2, 3]       + [0, inf])            * [2, 4]
    e1 = (_var((-2, 3)) + _var((0, None))) * 2 * _var((1, 2))
    #  = [-2, inf] * [2, 4]

    assert expr_bounds(e1) == Bound(-8, None)
    assert expr_bounds(3 - _var((0, 2))) == Bound(1, 3)


def test_bound_abs():
    e0 = abs(_var((-10, -1)))
    assert expr_bounds(e0) == Bound(1, 10)

    e1 = abs(_var((1, 10)))
    assert expr_bounds(e1) == Bound(1, 10)

    e2 = abs(_var((-10, 20)))
    assert expr_bounds(e2) == Bound(0, 20)


def test_bound_sqrt():
    with pytest.raises(ValueError):
        expr_bounds(aml.sqrt(_var()))
    assert expr_bounds(aml.sqrt(_var((0, None)))) == Bound(0, None)
    assert expr_bounds(aml.sqrt(_var((0, 2)))) == Bound(0, np.sqrt(2))


def test_bound_log():
    with pytest.raises(ValueError):
        expr_bounds(aml.log(_var((0, None))))
    assert expr_bounds(aml.log(_var((1, None)))) == Bound(0, None)
    assert expr_bounds(aml.log(_var((1, 2)))) == Bound(0, np.log(2))


def test_bound_asin():
    with pytest.raises(ValueError):
        expr_bounds(aml.asin(_var((0, None))))
    assert expr_bounds(aml.asin(_var((-1, 1)))) == Bound(-np.pi/2, np.pi/2)


def test_bound_acos():
    with pytest.raises(ValueError):
        expr_bounds(aml.acos(_var((0, None))))
    assert expr_bounds(aml.acos(_var((-1, 1)))) == Bound(0, np.pi)


def test_bound_atan():
    assert expr_bounds(aml.atan(_var())) == Bound(-np.pi/2, np.pi/2)


def test_bound_exp():
    assert expr_bounds(aml.exp(_var())) == Bound(0, None)
    assert expr_bounds(aml.exp(_var((0, None)))) == Bound(1, None)


def test_bound_sin():
    def assert_sin_bound(bounds, expected_bounds):
        assert expr_bounds(aml.sin(_var(bounds))) == Bound(*expected_bounds)

    assert_sin_bound((None, None), (-1, 1))
    assert_sin_bound((0, np.pi), (0, 1))
    assert_sin_bound((-np.pi, 0), (-1, 0))
    assert_sin_bound(
        (0.5 * np.pi - 0.5, 0.5 * np.pi + 0.5),
        (np.sin(0.5 * np.pi - 0.5), 1)
    )
    assert_sin_bound(
        (np.pi - 0.5, np.pi + 0.5),
        (np.sin(np.pi + 0.5), np.sin(np.pi - 0.5))
    )
    assert_sin_bound(
        (2 * np.pi - 0.1, 2 * np.pi + 0.5),
        (np.sin(2*np.pi - 0.1), np.sin(2*np.pi + 0.5))
    )


def test_bound_cos():
    def assert_cos_bound(bounds, expected_bounds):
        assert expr_bounds(aml.cos(_var(bounds))) == Bound(*expected_bounds)

    assert_cos_bound((None, None), (-1, 1))
    assert_cos_bound((0, np.pi), (-1, 1))
    assert_cos_bound((-0.5*np.pi, 0.5*np.pi), (0, 1))
    assert_cos_bound((0, 0.5*np.pi), (0, 1))
    assert_cos_bound((np.pi - 0.2, 2*np.pi), (-1, 1))
    assert is_nonnegative(aml.cos(_var((1.5*np.pi, 2.5*np.pi))))


def test_bound_tan():
    def assert_tan_bound(bounds, expected_bounds):
        assert expr_bounds(aml.tan(_var(bounds))) == Bound(*expected_bounds)

    assert_tan_bound((None, None), (None, None))
    assert_tan_bound((0, 0.5 * np.pi), (0, None))
    assert_tan_bound((0.5*np.pi, 0.5*np.pi), (None, None))
    assert_tan_bound((np.pi-0.1, np.pi+0.1), (-np.tan(0.1), np.tan(0.1)))


def test_bound_negation():
    e0 = -_var((-10, 1))
    assert expr_bounds(e0) == Bound(-1, 10)

    e1 = -abs(_var())
    assert expr_bounds(e1) == Bound(None, 0)


def test_is_positive():
    e0 = _var() + 3
    assert not is_positive(e0)

    e1 = abs(_var((-8, -3)))
    assert is_positive(e1)

    e2 = abs(_var() * 3.0 - 20)
    assert is_nonnegative(e2)

    assert is_nonnegative(aml.sin(_var((0, np.pi))))
    assert is_nonpositive(aml.cos(_var((0.5*np.pi, 1.5*np.pi))))

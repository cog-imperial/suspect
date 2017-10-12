import pytest
import pyomo.environ as aml
from convexity_detection.linearity import *
from convexity_detection.math import pi
from util import _var


def test_monotonicity_enum():
    assert Monotonicity.Nondecreasing.is_nondecreasing()
    assert Monotonicity.Nonincreasing.is_nonincreasing()
    assert Monotonicity.Constant.is_nondecreasing()
    assert Monotonicity.Constant.is_nonincreasing()
    assert Monotonicity.Constant.is_constant()


def test_is_constant():
    x = _var()
    y = _var()

    assert is_constant(aml.log(2.0))
    assert not is_constant(aml.log(x))
    x.fix(2.0)
    assert not is_constant(x + aml.tan(y))
    y.fix(3.0)
    assert is_constant(x + aml.sin(y/x))


def test_is_linear():
    x = _var()

    variables = [aml.Var() for _ in range(10)]
    for v in variables:
        v.construct()

    e0 = sum(v for v in variables)
    assert is_linear(e0)

    e1 = sum(x*v for v in variables)
    assert not is_linear(e1)

    x.fix(2.0)
    assert is_linear(e1)

    for v in variables:
        v.fix(1.0)
    assert is_linear(e1)


def test_monotonicity_linear():
    x = _var()
    y = _var()
    z = _var((0, None))

    assert is_unknown(x - y)
    assert is_nondecreasing(x + 10)
    assert is_nondecreasing(x + 10*y)
    assert is_nondecreasing(x + 0*y)
    assert is_nonincreasing(-x - y)
    assert is_nondecreasing(x + y + z)


def test_monotonicity_product():
    x = _var()
    y = _var()
    z = _var()

    assert is_unknown(-3*x*y*z)

    x = _var((0, None))
    y = _var((0, None))
    assert is_nondecreasing(x*y)

    x = _var((None, 0))
    y = _var((None, 0))
    assert is_nonincreasing(x*y)

    x = _var((None, 0))
    y = _var((None, 0))
    assert is_nondecreasing(-x*y)


def test_monotonicity_division():
    x = _var()
    y = _var()

    assert is_unknown(1/x)
    assert is_unknown(x/y)

    y = _var((0, None))
    assert is_nonincreasing(1/y)
    assert is_nondecreasing(-1/y)


def test_monotonicity_abs():
    x = _var()
    y = _var((0, None))
    z = _var((None, 0))

    assert is_unknown(abs(x))
    assert is_nondecreasing(abs(y))
    assert is_nonincreasing(abs(-z))


def test_monotonicity_negation():
    x = _var()
    y = _var((0, None))
    z = _var((None, 0))

    assert is_unknown(-abs(x))
    assert is_nonincreasing(-abs(y))
    assert is_nondecreasing(-abs(-z))


def test_monotonicity_sqrt():
    x = _var((0, None))
    y = _var((0, 2))

    assert is_nondecreasing(aml.sqrt(x))
    assert is_nonincreasing(aml.sqrt(2 - y))


def test_monotonicity_log():
    x = _var((1, None))
    y = _var((0, 2))

    assert is_nondecreasing(aml.log(x))
    assert is_nonincreasing(aml.log(3 - y))


def test_monotonicity_asin():
    x = _var((-1, 1))
    y = _var((0, 2))

    assert is_nondecreasing(aml.asin(x))
    assert is_nonincreasing(aml.asin(1 - y))


def test_monotonicity_atan():
    x = _var()
    y = _var()

    assert is_nondecreasing(aml.atan(x))
    assert is_nonincreasing(aml.atan(1 - y))


def test_monotonicity_tan():
    x = _var()
    y = _var()

    assert is_nondecreasing(aml.tan(x))
    assert is_nonincreasing(aml.tan(1 - y))


def test_monotonicity_exp():
    x = _var()
    y = _var()

    assert is_nondecreasing(aml.exp(x))
    assert is_nonincreasing(aml.exp(1 - y))


def test_monotonicity_acos():
    x = _var((-1, 1))
    y = _var((0, 2))

    assert is_nonincreasing(aml.acos(x))
    assert is_nondecreasing(aml.acos(1 - y))


def test_monotonicity_sin():
    assert is_unknown(aml.sin(_var()))
    assert is_nondecreasing(aml.sin(_var((-0.5*pi, 0.5*pi))))
    assert is_nondecreasing(aml.sin(_var((0, 0.5*pi))))
    assert is_nonincreasing(aml.sin(_var((2.5*pi, 3.5*pi))))
    assert is_nondecreasing(aml.sin(_var((1.5*pi, 2.5*pi))))
    assert is_unknown(aml.sin(_var((0.5*pi-0.2, 0.5*pi + 0.1))))


def test_monotonicity_cos():
    assert is_unknown(aml.cos(_var()))
    assert is_nondecreasing(aml.cos(_var((-pi, 0))))
    assert is_nonincreasing(aml.cos(_var((2*pi, 3*pi))))
    assert is_unknown(aml.cos(_var((-0.01, 0.01))))

import pytest
import pyomo.environ as aml
from convexity_detection.linearity import *
from convexity_detection.pyomo_compat import *
import numpy as np


set_pyomo4_expression_tree()


def test_is_constant():
    x = aml.Var()
    x.construct()
    y = aml.Var()
    y.construct()

    assert is_constant(aml.log(2.0))
    assert not is_constant(aml.log(x))
    x.fix(2.0)
    assert not is_constant(x + aml.tan(y))
    y.fix(3.0)
    assert is_constant(x + aml.sin(y/x))


def test_is_linear():
    x = aml.Var()
    x.construct()

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


def test_is_nondecreasing():
    x = aml.Var()
    x.construct()

    e0 = x - 2
    assert is_nondecreasing(e0)

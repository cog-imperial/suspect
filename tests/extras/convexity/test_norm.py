# pylint: skip-file
import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from tests.conftest import PlaceholderExpression as PE
from tests.strategies import reals
from suspect.expression import ExpressionType as ET, UnaryFunctionType as UFT
from suspect.interval import Interval as I
from suspect.convexity import Convexity
from suspect.extras.convexity.norm import L2NormRule
from suspect.math import almosteq
from mpmath import mp


@st.composite
def constants(draw):
    return PE(ET.Constant)


@st.composite
def powers(draw):
    base = PE(ET.Variable)
    expo = PE(ET.Constant, value=2.0)
    return PE(ET.Power, [base, expo])


@st.composite
def products(draw):
    x = PE(ET.Variable)
    return PE(ET.Product, [x, x])


@pytest.mark.skip('Not updated')
@given(st.lists(st.one_of(constants(), powers(), products()), min_size=1))
def test_l2_norm(children):
    rule = L2NormRule()
    sum_ = PE(ET.Sum, children)
    result = rule.checked_apply(PE(ET.UnaryFunction, [sum_], func_type=UFT.Sqrt), None)
    assert result.is_convex()

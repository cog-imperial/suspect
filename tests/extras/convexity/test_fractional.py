# pylint: skip-file
import pytest
from hypothesis import given, assume
from tests.conftest import PlaceholderExpression as PE
from tests.strategies import reals
from suspect.expression import ExpressionType as ET
from suspect.interval import Interval as I
from suspect.convexity import Convexity
from suspect.extras.convexity.fractional import FractionalRule
from suspect.math import almosteq
from mpmath import mp

MATH_PRECISION = 50
# mp.dps = MATH_PRECISION
MAX_NUM = 10**(MATH_PRECISION//2)

# TEST SETUP
#
# We want to test convexity of
#
# f(x) = (a_1 x + b_1) / (a_2 x + b_2)
#
# its second derivative is
#
# f''(x) = -2 a_2 ( (a_1 b_2 - a_2 b_1) / (a_2 x + b_2)^3 )
#
# So the sign of f''(x) depends on a_2, (a_1 b_2 - a_2 b_1) and (a_2 x + b_2)
#
# We test with denominator/denominator linear, constant and variable.

class FractionalContext:
    def __init__(self, bounds=None):
        if bounds is None:
            bounds = {}
        self._b = bounds

    def bounds(self, expr):
        return self._b[expr]


@pytest.mark.skip('Not update')
# (a_1 x) / x is linear (constant)
@given(coef=reals(allow_infinity=False))
def test_linear_over_variable(coef):
    rule = FractionalRule()
    x = PE(ET.Variable)
    num = PE(ET.Linear, [x], coefficients=[coef], constant_term=0.0)
    ctx = FractionalContext({
        x: I(0, None),
    })
    result = rule.apply(PE(ET.Division, [num, x]), ctx)
    assert result == Convexity.Linear


@pytest.mark.skip('Not update')
# (a_1 x + b_1) / x
@given(coef=reals(allow_infinity=False), const=reals(allow_infinity=False))
def test_linear_with_constant_over_variable(coef, const):
    assume(coef != 0.0)
    assume(coef < MAX_NUM and const < MAX_NUM)

    rule = FractionalRule()
    x = PE(ET.Variable)
    num = PE(ET.Linear, [x], coefficients=[coef], constant_term=const)
    ctx = FractionalContext({
        x: I(0, None),
    })
    result = rule.apply(PE(ET.Division, [num, x]), ctx)
    if almosteq(const, 0):
        expected = Convexity.Linear
    elif const > 0:
        expected = Convexity.Convex
    elif const < 0:
        expected = Convexity.Concave
    # assert result == expected

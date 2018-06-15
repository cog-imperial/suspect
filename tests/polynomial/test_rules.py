# pylint: skip-file
import pytest
from hypothesis import given
import hypothesis.strategies as st
from tests.conftest import PlaceholderExpression as PE
from suspect.expression import ExpressionType as ET
from suspect.polynomial.degree import PolynomialDegree
from suspect.polynomial.rules import *


@st.composite
def polynomial_degrees(draw, allow_not_polynomial=True):
    if allow_not_polynomial:
        degree = draw(st.one_of(st.none(), st.integers(min_value=0)))
    else:
        degree = draw(st.integers(min_value=0))
    return PolynomialDegree(degree)


@st.composite
def placeholder_expressions(draw):
    return PE(draw(st.sampled_from(ET)))


def test_variable_is_always_1():
    rule = VariableRule()
    result = rule.checked_apply(PE(ET.Variable), None)
    assert result.degree == 1


def test_constant_is_always_0():
    rule = ConstantRule()
    result = rule.checked_apply(PE(ET.Constant), None)
    assert result.degree == 0


@given(polynomial_degrees())
@pytest.mark.parametrize('rule_cls,root_type', [
    (ConstraintRule ,ET.Constraint),
    (ObjectiveRule, ET.Objective),
    (NegationRule, ET.Negation),
])
def test_passtrough_rules(rule_cls, root_type, child_degree):
    rule = rule_cls()
    child = PE(ET.Variable)
    ctx = {child: child_degree}
    result = rule.checked_apply(PE(root_type, [child]), ctx)
    assert result == child_degree


@given(polynomial_degrees())
def test_division_rule_with_constant_denominator(child_degree):
    rule = DivisionRule()
    num = PE(ET.Variable)
    den = PE(ET.Variable)
    ctx = {
        num: child_degree,
        den: PolynomialDegree(0),
    }
    result = rule.checked_apply(PE(ET.Division, [num, den]), ctx)
    assert result == child_degree


@given(polynomial_degrees())
def test_division_rule_with_nonconstant_denominator(child_degree):
    rule = DivisionRule()
    num = PE(ET.Variable)
    den = PE(ET.Variable)
    ctx = {
        num: child_degree,
        den: PolynomialDegree(1),
    }
    result = rule.checked_apply(PE(ET.Division, [num, den]), ctx)
    assert not result.is_polynomial()


@given(st.lists(st.tuples(polynomial_degrees(), placeholder_expressions()), min_size=1))
def test_product_rule(children_degrees):
    children = []
    ctx = {}
    expected = 0
    for degree, child in children_degrees:
        children.append(child)
        ctx[child] = degree
        if expected is None:
            continue
        if degree.is_polynomial():
            expected += degree.degree
        else:
            expected = None
    expected = PolynomialDegree(expected)
    rule = ProductRule()
    result = rule.checked_apply(PE(ET.Product, children), ctx)
    assert result == expected


def test_linear_rule_with_no_children():
    rule = LinearRule()
    result = rule.checked_apply(PE(ET.Linear, []), None)
    assert result.degree == 0


@given(st.integers(min_value=1))
def test_linear_rule_with_children(children_size):
    rule = LinearRule()
    children = [PE(ET.Variable)] * children_size
    result = rule.checked_apply(PE(ET.Linear, children), None)
    assert result.degree == 1


@given(st.lists(st.tuples(polynomial_degrees(), placeholder_expressions()), min_size=1))
def test_sum_rule(children_degrees):
    children = []
    ctx = {}
    expected = 0
    for degree, child in children_degrees:
        children.append(child)
        ctx[child] = degree
        if expected is None:
            continue
        if degree.is_polynomial() and degree.degree > expected:
            expected = degree.degree
        if not degree.is_polynomial():
            expected = None

    expected = PolynomialDegree(expected)
    rule = SumRule()
    result = rule.checked_apply(PE(ET.Sum, children), ctx)
    assert result == expected


def test_power_with_non_polynomial_exponent():
    ctx = {}
    base = PE(ET.Variable)
    ctx[base] = PolynomialDegree(1)
    expo = PE(ET.UnaryFunction)
    ctx[expo] = PolynomialDegree(None)
    rule = PowerRule()
    result = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert not result.is_polynomial()


def test_power_with_non_polynomial_base():
    ctx = {}
    base = PE(ET.UnaryFunction)
    ctx[base] = PolynomialDegree(None)
    expo = PE(ET.Variable)
    ctx[expo] = PolynomialDegree(1)
    rule = PowerRule()
    result = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert not result.is_polynomial()


def test_power_constant_power_constant():
    ctx = {}
    base = PE(ET.Constant)
    ctx[base] = PolynomialDegree(0)
    expo = PE(ET.Constant)
    ctx[expo] = PolynomialDegree(0)
    rule = PowerRule()
    result = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert result.is_polynomial() and result.degree == 0


def test_power_non_constant():
    ctx = {}
    base = PE(ET.Variable)
    ctx[base] = PolynomialDegree(1)
    expo = PE(ET.Constant)
    ctx[expo] = PolynomialDegree(0)
    rule = PowerRule()
    result = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert not result.is_polynomial()


def test_power_non_constant_polynomial():
    ctx = {}
    base = PE(ET.Variable)
    ctx[base] = PolynomialDegree(1)
    expo = PE(ET.Product)
    ctx[expo] = PolynomialDegree(4)
    rule = PowerRule()
    result = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert not result.is_polynomial()


def test_power_constant_value():
    ctx = {}
    base = PE(ET.Product)
    ctx[base] = PolynomialDegree(2)
    expo = PE(ET.Constant, is_constant=True, value=4.0)
    ctx[expo] = PolynomialDegree(0)
    rule = PowerRule()
    result = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert result.is_polynomial() and result.degree == 8


def test_unary_function():
    rule = UnaryFunctionRule()
    ctx = {}
    result = rule.checked_apply(PE(ET.UnaryFunction), ctx)
    assert not result.is_polynomial()

# pylint: skip-file
import hypothesis.strategies as st
import pyomo.environ as pe
import pytest
from hypothesis import given, assume, settings, HealthCheck
from pyomo.core.kernel.component_map import ComponentMap

from suspect.polynomial.rules import *
from suspect.polynomial.visitor import PolynomialDegreeVisitor
from suspect.pyomo.expressions import (
    MonomialTermExpression,
    DivisionExpression,
    SumExpression,
    Constraint,
    Objective,
    PowExpression,
)
from tests.strategies import expressions, variables, reals, unary_function_types


@pytest.fixture(scope='module')
def visitor():
    return PolynomialDegreeVisitor()


@st.composite
def polynomial_degrees(draw, allow_not_polynomial=True):
    if allow_not_polynomial:
        degree = draw(st.one_of(st.none(), st.integers(min_value=0)))
    else:
        degree = draw(st.integers(min_value=0))
    return PolynomialDegree(degree)


def test_variable_is_always_1(visitor):
    matched, result = visitor.visit_expression(pe.Var(), None)
    assert matched
    assert result.degree == 1


def test_constant_is_always_0(visitor):
    matched, result = visitor.visit_expression(123.0, None)
    assert matched
    assert result.degree == 0


@given(expressions(), polynomial_degrees())
def test_objective(visitor, child, child_degree):
    expr = Objective('obj', children=[child])
    poly = ComponentMap()
    poly[child] = child_degree
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert result == child_degree


@given(expressions(), polynomial_degrees())
def test_constraint(visitor, child, child_degree):
    expr = Constraint('cons', 0.0, None, children=[child])
    poly = ComponentMap()
    poly[child] = child_degree
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert result == child_degree


@given(expressions(), polynomial_degrees())
def test_negation(visitor, child, child_degree):
    expr = -child
    assume(child.is_expression_type() and not isinstance(expr, MonomialTermExpression))
    poly = ComponentMap()
    poly[child] = child_degree
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert result == child_degree


def test_division_rule_with_constant_denominator(visitor):
    child = pe.Var()
    num = 1.0
    poly = ComponentMap()
    poly[num] = PolynomialDegree(0)
    poly[child] = PolynomialDegree(0)
    expr = DivisionExpression([num, child])
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert result.is_constant()


@given(polynomial_degrees())
def test_division_rule_with_nonconstant_denominator(visitor, den_degree):
    assume(not den_degree.is_constant())
    child = pe.Var()
    num = 1.0
    expr = DivisionExpression([num, child])
    poly = ComponentMap()
    poly[num] = PolynomialDegree(0)
    poly[child] = den_degree
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert not result.is_polynomial()



@given(expressions(), expressions(), polynomial_degrees(), polynomial_degrees())
def test_product_rule(visitor, f, g, poly_f, poly_g):
    expr = f * g
    poly = ComponentMap()
    poly[f] = poly_f
    poly[g] = poly_g
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    if poly_f.is_polynomial() and poly_g.is_polynomial():
        assert result.is_polynomial()
        assert result.degree == poly_f.degree + poly_g.degree
    else:
        assert not result.is_polynomial()


@pytest.mark.skip('Linear Expression not implemented.')
@given(st.integers(min_value=1, max_value=100))
def test_linear_rule_with_children(children_size):
    rule = LinearRule()
    children = [PE(ET.Variable)] * children_size
    result = rule.apply(PE(ET.Linear, children), None)
    assert result.degree == 1


@settings(suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(st.tuples(expressions(), polynomial_degrees()), min_size=2, max_size=4))
def test_sum_rule(visitor, data):
    children = [c for c, _ in data]
    for c in children:
        assume(c.is_expression_type() and not isinstance(c, SumExpression))
    degrees = [d for _, d in data]
    poly = ComponentMap()
    for c, d in data:
        poly[c] = d
    expr = sum(children)
    print(expr, expr.args)
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    if all(d.is_polynomial() for d in degrees):
        assert result.is_polynomial()
        assert result.degree == max(d.degree for d in degrees)
    else:
        assert not result.is_polynomial()


@given(expressions(), expressions(), polynomial_degrees())
def test_power_with_non_polynomial_exponent(visitor, base, expo, base_poly):
    poly = ComponentMap()
    poly[base] = base_poly
    poly[expo] = PolynomialDegree(None)
    expr = base ** expo
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert not result.is_polynomial()


@given(expressions(), expressions(), polynomial_degrees())
def test_power_with_non_polynomial_base(visitor, base, expo, expo_poly):
    poly = ComponentMap()
    poly[base] = PolynomialDegree(None)
    poly[expo] = expo_poly
    expr = base ** expo
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert not result.is_polynomial()


@given(variables(), reals(min_value=1.1))
def test_power_constant_power_constant(visitor, v, c):
    expr = v ** c
    poly = ComponentMap()
    poly[v] = PolynomialDegree(0)
    poly[c] = PolynomialDegree(0)
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert result.is_polynomial() and result.degree == 0


@given(variables(), reals(allow_infinity=False))
def test_power_non_constant(visitor, v, c):
    assume(c != int(c))
    expr = v ** c
    poly = ComponentMap()
    poly[v] = PolynomialDegree(1)
    poly[c] = PolynomialDegree(0)
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert not result.is_polynomial()


@given(variables(), expressions())
def test_power_non_constant_polynomial(visitor, v, c):
    expr = v ** c
    poly = ComponentMap()
    poly[v] = PolynomialDegree(1)
    poly[c] = PolynomialDegree(4)
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert not result.is_polynomial()


@given(expressions(), st.integers(min_value=1), st.integers(min_value=2))
def test_power_constant_value(visitor, base, base_poly, expo):
    expr = base ** expo
    assume(isinstance(expr, PowExpression))
    poly = ComponentMap()
    poly[base] = PolynomialDegree(base_poly)
    poly[expo] = PolynomialDegree(0)
    matched, result = visitor.visit_expression(expr, poly)
    assert matched
    assert result.is_polynomial() and result.degree == base_poly * expo


@given(expressions(), polynomial_degrees(), unary_function_types)
def test_unary_function(visitor, child, poly_child, func_name):
    if func_name == 'abs':
        expr = abs(child)
        poly = ComponentMap()
        poly[child] = poly_child

        matched, result = visitor.visit_expression(expr, poly)
        assert matched
        assert result == poly_child

    else:
        expr = getattr(pe, func_name)(child)

        matched, result = visitor.visit_expression(expr, None)
        assert matched
        assert not result.is_polynomial()

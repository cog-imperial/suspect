# pylint: skip-file
import pytest
from hypothesis import given, assume, reproduce_failure
from hypothesis.strategies import integers, lists
import pyomo.environ as pe
from pyomo.core.kernel.component_map import ComponentMap
from suspect.pyomo.expressions import *
from suspect.fbbt.propagation.rules import *
from suspect.fbbt.propagation.visitor import BoundsPropagationVisitor
from suspect.interval import Interval
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
    intervals,
)


@pytest.fixture
def visitor():
    return BoundsPropagationVisitor()


@given(reals(), reals())
def test_variable_bound(visitor, a, b):
    lb = min(a, b)
    ub = max(a, b)
    var = pe.Var(bounds=(lb, ub))
    var.construct()
    rule = VariableRule()
    assert rule.apply(var, None) == Interval(lb, ub)


@given(reals(), reals())
def test_variable_visito(visitor, a, b):
    lb = min(a, b)
    ub = max(a, b)
    var = pe.Var(bounds=(lb, ub))
    var.construct()
    bounds = ComponentMap()
    visitor.visit(var, bounds)
    assert bounds[var] == Interval(lb, ub)


@given(reals())
def test_constant_bound(c):
    const = NumericConstant(c)
    rule = ConstantRule()
    assert rule.apply(const, None) == Interval(c, c)


@given(reals())
def test_constant_visitor(visitor, c):
    const = NumericConstant(c)
    rule = ConstantRule()
    bounds = ComponentMap()
    visitor.visit(const, bounds)
    assert bounds[const] == Interval(c, c)


@given(reals())
def test_constant_of_float_bound(const):
    rule = ConstantRule()
    assert rule.apply(const, None) == Interval(const, const)


@given(reals())
def test_constant_of_float_visitor(visitor, const):
    bounds = ComponentMap()
    visitor.visit(const, bounds)


@given(reals(), reals(), reals(), reals())
def test_constraint_bound(a, b, c, d):
    e_lb, e_ub = min(a, b), max(a, b)
    c_lb, c_ub = min(c, d), max(c, d)
    assume(max(e_lb, c_lb) <= min(e_ub, c_ub))
    p = PlaceholderExpression()
    cons = Constraint('c0', lower_bound=c_lb, upper_bound=c_ub, children=[p])
    bounds = ComponentMap()
    bounds[p] = Interval(e_lb, e_ub)
    rule = ConstraintRule()
    expected = Interval(max(e_lb, c_lb), min(c_ub, e_ub))
    assert rule.apply(cons, bounds) == expected


@given(reals(max_value=100.0), reals(max_value=100.0))
def test_objective_bound(a, b):
    lb, ub = min(a, b), max(a, b)
    assume(lb < ub)
    p = PlaceholderExpression()
    o = Objective('obj', children=[p])
    bounds = ComponentMap()
    bounds[p] = Interval(lb, ub)
    rule = ObjectiveRule()
    assert rule.apply(o, bounds) == Interval(lb, ub)


# pyomo has binary products
@given(lists(intervals(), min_size=2, max_size=2))
def test_product_visitor(visitor, children_bounds):
    children = [PlaceholderExpression() for _ in children_bounds]
    bounds = ComponentMap()
    expected = 1.0
    for child, child_bound in zip(children, children_bounds):
        bounds[child] = child_bound
        expected *= child_bound

    expr = ProductExpression(children)
    visitor.visit(expr, bounds)
    assert bounds[expr] == expected


@given(lists(intervals(), min_size=2))
def test_product_bound(children_bounds):
    children = [PlaceholderExpression() for _ in children_bounds]
    bounds = ComponentMap()
    expected = 1.0
    for child, child_bound in zip(children, children_bounds):
        bounds[child] = child_bound
        expected *= child_bound

    expr = PlaceholderExpression(children=children)
    rule = ProductRule()
    rule.apply(expr, bounds) == expected


@pytest.mark.parametrize('bound,expected', [
    (Interval(-1, 1), Interval(None, None)),
    (Interval(2, 10), Interval(0.1, 0.5)),
])
def test_reciprocal_bound(bound, expected):
    child = PlaceholderExpression()
    expr = PlaceholderExpression(children=[child])
    bounds = ComponentMap()
    bounds[child] = bound
    rule = ReciprocalRule()
    assert rule.apply(expr, bounds) == expected


@pytest.mark.parametrize('bound,expected', [
    (Interval(-1, 1), Interval(None, None)),
    (Interval(2, 10), Interval(0.1, 0.5)),
])
def test_reciprocal_visitor(visitor, bound, expected):
    child = PlaceholderExpression()
    expr = ReciprocalExpression((child,))
    bounds = ComponentMap()
    bounds[child] = bound
    visitor.visit(expr, bounds)
    assert bounds[expr] == expected


@given(intervals())
def test_negation_bound(child_bounds):
    child = PlaceholderExpression()
    expr = PlaceholderExpression(children=[child])
    bounds = ComponentMap()
    bounds[child] = child_bounds
    rule = NegationRule()
    assert - rule.apply(expr, bounds) == child_bounds


@given(integers(min_value=1))
def test_even_pow_bound(n):
    p = PlaceholderExpression()
    c = NumericConstant(2*n)
    pow_ = PowExpression([p, c])
    bounds = ComponentMap()
    rule = PowerRule()
    assert rule.apply(pow_, bounds).is_nonnegative()


@given(integers(min_value=1))
def test_even_pow_float_exponent_bound(n):
    p = PlaceholderExpression()
    c = 2 * n
    pow_ = PowExpression([p, c])
    bounds = ComponentMap()
    rule = PowerRule()
    assert rule.apply(pow_, bounds).is_nonnegative()


@given(integers(min_value=1))
def test_even_pow_non_const_bound(n):
    p = PlaceholderExpression()
    c = PlaceholderExpression(is_constant=False)
    pow_ = PowExpression([p, c])
    bounds = ComponentMap()
    rule = PowerRule()
    assert rule.apply(pow_, bounds) == Interval(None, None)


@given(integers(min_value=1))
def test_even_pow_odd_bound(n):
    p = PlaceholderExpression()
    c = 2 * n + 1
    pow_ = PowExpression([p, c])
    bounds = ComponentMap()
    rule = PowerRule()
    assert rule.apply(pow_, bounds) == Interval(None, None)


@given(integers(min_value=1))
def test_even_pow_negative_bound(n):
    p = PlaceholderExpression()
    c = -2 * n
    pow_ = PowExpression([p, c])
    bounds = ComponentMap()
    rule = PowerRule()
    assert rule.apply(pow_, bounds) == Interval(None, None)


@pytest.mark.parametrize('rule_cls,func_name,interval,expected', [
    (AbsRule, 'abs', Interval(-10, 3), Interval(0, 10)),
    (SqrtRule, 'sqrt', Interval(4, 16), Interval(2, 4)),
    (ExpRule, 'exp', Interval(0, 1).log(), Interval(0, 1)),
    (LogRule, 'log', Interval(0, 1).exp(), Interval(0, 1)),
    (SinRule, 'sin', Interval(0, 1).asin(), Interval(0, 1)),
    (CosRule, 'cos', Interval(0, 1).acos(), Interval(0, 1)),
    (TanRule, 'tan', Interval(0, 1).atan(), Interval(0, 1)),
    (AsinRule, 'asin', Interval(0, 1).sin(), Interval(0, 1)),
    (AcosRule, 'acos', Interval(0, 1).cos(), Interval(0, 1)),
    (AtanRule, 'atan', Interval(0, 1).tan(), Interval(0, 1)),
])
def test_unary_function(rule_cls, func_name, interval, expected):
    child = PlaceholderExpression()
    expr = PlaceholderExpression(children=[child])
    rule = rule_cls()
    bounds = ComponentMap()
    bounds[child] = interval
    assert rule.apply(expr, bounds) == expected



def test_visitor_tightens_new_bounds(visitor):
    bounds = ComponentMap()
    expr = PlaceholderExpression()
    assert bounds.get(expr, None) is None

    assert visitor.handle_result(expr, Interval(0, 2), bounds)
    assert bounds[expr] == Interval(0, 2)

    assert not visitor.handle_result(expr, Interval(0, 2), bounds)
    assert bounds[expr] == Interval(0, 2)

    assert visitor.handle_result(expr, Interval(0, 1), bounds)
    assert bounds[expr] == Interval(0, 1)

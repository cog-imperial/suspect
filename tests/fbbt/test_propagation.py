# pylint: skip-file
import pyomo.environ as pe
import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers, lists
from pyomo.core.kernel.component_map import ComponentMap

from suspect.fbbt.propagation.rules import *
from suspect.fbbt.propagation.visitor import BoundsPropagationVisitor
from suspect.interval import Interval
from suspect.pyomo.expressions import *
from tests.strategies import expressions, reals, intervals


@pytest.fixture(scope='module')
def visitor():
    return BoundsPropagationVisitor()


@given(reals(), reals())
def test_variable_bound(visitor, a, b):
    lb = min(a, b)
    ub = max(a, b)
    var = pe.Var(bounds=(lb, ub))
    var.construct()
    matched, result = visitor.visit_expression(var, None)
    assert matched
    assert result == Interval(lb, ub)


@given(reals())
def test_constant_bound(visitor, c):
    const = NumericConstant(c)
    matched, result = visitor.visit_expression(const, None)
    assert matched
    assert result == Interval(c, c)


@given(reals())
def test_constant_of_float_bound(visitor, const):
    rule = ConstantRule()
    matched, result = visitor.visit_expression(const, None)
    assert result == Interval(const, const)


@given(reals(), reals(), reals(), reals())
def test_constraint_bound(visitor, a, b, c, d):
    e_lb, e_ub = min(a, b), max(a, b)
    c_lb, c_ub = min(c, d), max(c, d)
    assume(max(e_lb, c_lb) <= min(e_ub, c_ub))
    child = pe.Var()

    cons = Constraint('c0', lower_bound=c_lb, upper_bound=c_ub, children=[child])
    bounds = ComponentMap()
    bounds[child] = Interval(e_lb, e_ub)
    matched, result = visitor.visit_expression(cons, bounds)
    expected = Interval(max(e_lb, c_lb), min(c_ub, e_ub))
    assert result == expected


@given(reals(max_value=100.0), reals(max_value=100.0))
def test_objective_bound(visitor, a, b):
    lb, ub = min(a, b), max(a, b)
    assume(lb < ub)
    child = pe.Var()
    o = Objective('obj', children=[child])
    bounds = ComponentMap()
    bounds[child] = Interval(lb, ub)
    matched, result = visitor.visit_expression(o, bounds)
    assert matched
    assert result == Interval(lb, ub)


# pyomo has binary products
@given(lists(intervals(), min_size=2, max_size=2))
def test_product_visitor(visitor, children_bounds):
    children = [pe.Var() for _ in children_bounds]
    bounds = ComponentMap()
    expected = 1.0
    for child, child_bound in zip(children, children_bounds):
        bounds[child] = child_bound
        expected *= child_bound

    expr = children[0] * children[1]
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert result == expected


@pytest.mark.parametrize('bound,expected', [
    (Interval(-1, 1), Interval(None, None)),
    (Interval(2, 10), Interval(0.1, 0.5)),
])
def test_division_bound(visitor, bound, expected):
    num = pe.Var()
    child = pe.Var()
    expr = num / child
    bounds = ComponentMap()
    bounds[num] = Interval(1.0, 1.0)
    bounds[child] = bound
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert result == expected


@given(expressions(), intervals())
def test_negation_bound(visitor, child, child_bounds):
    expr = -child
    assume(child.is_expression_type())
    assume(expr.is_expression_type() and not isinstance(expr, MonomialTermExpression))
    bounds = ComponentMap()
    bounds[child] = child_bounds
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert -result == child_bounds


@given(expressions(), integers(min_value=1))
def test_even_pow_bound(visitor, base, n):
    c = NumericConstant(2*n)
    pow_ = base ** c
    bounds = ComponentMap()
    matched, result = visitor.visit_expression(pow_, bounds)
    assert matched
    assert result.is_nonnegative()


@given(expressions(), expressions())
def test_even_pow_non_const_bound(visitor, base, expo):
    assume(not expo.is_constant())
    expr = base ** expo
    bounds = ComponentMap()
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert result == Interval(None, None)


@given(expressions(), integers(min_value=1))
def test_even_pow_odd_bound(visitor, base, n):
    c = 2 * n + 1
    expr = base ** c
    bounds = ComponentMap()
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert result == Interval(None, None)


@given(expressions(), integers(min_value=1))
def test_even_pow_negative_bound(visitor, base, n):
    c = -2 * n
    expr = base ** c
    bounds = ComponentMap()
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert result == Interval(None, None)


@pytest.mark.parametrize('func_name,interval,expected', [
    ('abs', Interval(-10, 3), Interval(0, 10)),
    ('sqrt', Interval(4, 16), Interval(2, 4)),
    ('exp', Interval(0, 1).log(), Interval(0, 1)),
    ('log', Interval(0, 1).exp(), Interval(0, 1)),
    ('sin', Interval(0, 1).asin(), Interval(0, 1)),
    ('cos', Interval(0, 1).acos(), Interval(0, 1)),
    ('tan', Interval(0, 1).atan(), Interval(0, 1)),
    ('asin', Interval(0, 1).sin(), Interval(0, 1)),
    ('acos', Interval(0, 1).cos(), Interval(0, 1)),
    ('atan', Interval(0, 1).tan(), Interval(0, 1)),
])
@given(child=expressions())
def test_unary_function(visitor, func_name, interval, expected, child):
    if func_name == 'abs':
        expr = abs(child)
    else:
        expr = getattr(pe, func_name)(child)
    bounds = ComponentMap()
    bounds[child] = interval
    matched, result = visitor.visit_expression(expr, bounds)
    assert matched
    assert result == expected


@given(expressions())
def test_visitor_tightens_new_bounds(visitor, expr):
    bounds = ComponentMap()
    assert bounds.get(expr, None) is None

    assert visitor.handle_result(expr, Interval(0, 2), bounds)
    assert bounds[expr] == Interval(0, 2)

    assert not visitor.handle_result(expr, Interval(0, 2), bounds)
    assert bounds[expr] == Interval(0, 2)

    assert visitor.handle_result(expr, Interval(0, 1), bounds)
    assert bounds[expr] == Interval(0, 1)

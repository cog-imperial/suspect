# pylint: skip-file
import pytest
from hypothesis import given, assume
from hypothesis.strategies import integers
from suspect.fbbt.propagation import BoundsPropagationVisitor
import suspect.dag.expressions as dex
from suspect.interval import Interval
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
)


class PropagationContext:
    def __init__(self, bounds=None):
        if bounds is None:
            bounds = {}
        self._bounds = bounds

    def bounds(self, expr):
        return self._bounds[expr]

    def get_bounds(self, expr):
        return self._bounds.get(expr)

    def set_bounds(self, expr, value):
        self._bounds[expr] = value


@pytest.fixture
def visitor():
    return BoundsPropagationVisitor()


@given(reals(), reals())
def test_variable_bound(visitor, a, b):
    lb = min(a, b)
    ub = max(a, b)
    var = dex.Variable('x0', lower_bound=lb, upper_bound=ub)
    ctx = PropagationContext()
    visitor.visit(var, ctx)
    assert ctx.bounds(var) == Interval(lb, ub)


@given(reals())
def test_constant_bound(visitor, c):
    const = dex.Constant(c)
    ctx = PropagationContext()
    visitor.visit(const, ctx)
    assert ctx.bounds(const) == Interval(c, c)


@given(reals(), reals(), reals(), reals())
def test_constraint_bound(visitor, a, b, c, d):
    e_lb, e_ub = min(a, b), max(a, b)
    c_lb, c_ub = min(c, d), max(c, d)
    assume(max(e_lb, c_lb) <= min(e_ub, c_ub))
    p = PlaceholderExpression()
    cons = dex.Constraint('c0', lower_bound=c_lb, upper_bound=c_ub, children=[p])
    ctx = PropagationContext({p: Interval(e_lb, e_ub)})
    visitor.visit(cons, ctx)
    expected = Interval(max(e_lb, c_lb), min(c_ub, e_ub))
    assert ctx.bounds(cons) == expected


@given(reals(max_value=100.0), reals(max_value=100.0))
def test_objective_bound(visitor, a, b):
    lb, ub = min(a, b), max(a, b)
    assume(lb < ub)
    p = PlaceholderExpression()
    o = dex.Objective('obj', children=[p])
    ctx = PropagationContext({p: Interval(lb, ub)})
    visitor.visit(o, ctx)
    assert ctx.bounds(o) == Interval(lb, ub)


@given(integers(min_value=1))
def test_even_pow_bound(visitor, ctx, n):
    p = PlaceholderExpression()
    c = dex.Constant(2*n)
    pow_ = dex.PowExpression(children=[p, c])
    ctx = PropagationContext()
    visitor.visit(pow_, ctx)
    assert ctx.bounds(pow_).is_nonnegative()

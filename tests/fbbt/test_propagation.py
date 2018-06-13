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


@pytest.fixture
def visitor():
    return BoundsPropagationVisitor()


@given(reals(), reals())
def test_variable_bound(visitor, a, b):
    lb = min(a, b)
    ub = max(a, b)
    var = dex.Variable('x0', lower_bound=lb, upper_bound=ub)
    ctx = {}
    visitor.visit(var, ctx)
    assert ctx[var] == Interval(lb, ub)


@given(reals())
def test_constant_bound(visitor, c):
    const = dex.Constant(c)
    ctx = {}
    visitor.visit(const, ctx)
    assert ctx[const] == Interval(c, c)


@given(reals(), reals(), reals(), reals())
def test_constraint_bound(visitor, a, b, c, d):
    e_lb, e_ub = min(a, b), max(a, b)
    c_lb, c_ub = min(c, d), max(c, d)
    assume(max(e_lb, c_lb) <= min(e_ub, c_ub))
    p = PlaceholderExpression()
    ctx = {}
    ctx[p] = Interval(e_lb, e_ub)
    cons = dex.Constraint('c0', lower_bound=c_lb, upper_bound=c_ub, children=[p])
    visitor.visit(cons, ctx)
    expected = Interval(max(e_lb, c_lb), min(c_ub, e_ub))
    assert ctx[cons] == expected


@given(reals(max_value=100.0), reals(max_value=100.0))
def test_objective_bound(visitor, a, b):
    lb, ub = min(a, b), max(a, b)
    assume(lb < ub)
    p = PlaceholderExpression()
    ctx = {}
    ctx[p] = Interval(lb, ub)
    o = dex.Objective('obj', children=[p])
    visitor.visit(o, ctx)
    assert ctx[o] == Interval(lb, ub)


@given(integers(min_value=1))
def test_even_pow_bound(visitor, ctx, n):
    p = PlaceholderExpression()
    c = dex.Constant(2*n)
    pow_ = dex.PowExpression(children=[p, c])
    ctx = {}
    visitor.visit(pow_, ctx)
    assert ctx[pow_].is_nonnegative()

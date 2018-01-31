# Copyright 2017 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from unittest.mock import MagicMock
from hypothesis import given, assume
import hypothesis.strategies as st
from suspect.bound.propagation import BoundsPropagationVisitor
import suspect.dag.expressions as dex
from suspect.bound import ArbitraryPrecisionBound as Bound
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
)

class MockBoundsPropagationVisitor(BoundsPropagationVisitor):
    pass


@pytest.fixture
def mock_bounds_visitor(ctx=None):
    if ctx is None:
        ctx = {}
    return MockBoundsPropagationVisitor(ctx)


@given(reals(), reals())
def test_variable_bound(mock_bounds_visitor, a, b):
    lb = min(a, b)
    ub = max(a, b)
    var = dex.Variable('x0', lower_bound=lb, upper_bound=ub)
    mock_bounds_visitor(var)
    assert mock_bounds_visitor.get(var) == Bound(lb, ub)


@given(reals())
def test_constant_bound(mock_bounds_visitor, c):
    const = dex.Constant(c)
    mock_bounds_visitor(const)
    assert mock_bounds_visitor.get(const) == Bound(c, c)


@given(reals(), reals(), reals(), reals())
def test_constraint_bound(mock_bounds_visitor, a, b, c, d):
    e_lb, e_ub = min(a, b), max(a, b)
    c_lb, c_ub = min(c, d), max(c, d)
    assume(max(e_lb, c_lb) <= min(e_ub, c_ub))
    p = PlaceholderExpression()
    mock_bounds_visitor.set(p, Bound(e_lb, e_ub))
    cons = dex.Constraint('c0', lower_bound=c_lb, upper_bound=c_ub, children=[p])
    mock_bounds_visitor(cons)
    expected = Bound(max(e_lb, c_lb), min(c_ub, e_ub))
    assert mock_bounds_visitor.get(cons) == expected


@given(reals(), reals())
def test_objective_bound(mock_bounds_visitor, a, b):
    lb, ub = min(a, b), max(a, b)
    p = PlaceholderExpression()
    mock_bounds_visitor.set(p, Bound(lb, ub))
    o = dex.Objective('obj', children=[p])
    mock_bounds_visitor(o)
    assert mock_bounds_visitor.get(o) == Bound(lb, ub)

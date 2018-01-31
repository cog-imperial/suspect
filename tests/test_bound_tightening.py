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
from suspect.bound.tightening import *
import suspect.dag.expressions as dex
from suspect.bound import ArbitraryPrecisionBound as Bound
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
)


class TestTightenVariableInLinearComponent(object):
    def test_tighten_single_variable(self):
        variable = dex.Variable('x0', -10, 10)
        linear = dex.LinearExpression(coefficients=[-0.5], children=[variable])
        constraint = dex.Constraint('c0', lower_bound=0, upper_bound=20, children=[linear])
        bounds = {
            id(constraint): Bound(0, 20),
            id(variable): Bound(-10, 10),
        }
        new_bound = tighten_variable_in_linear_component(linear, [], variable, constraint, bounds)
        assert new_bound == Bound(-10, 0)

    def test_tighten_multiple_linear_variables(self):
        variable = dex.Variable('x0', -10, 10)
        bounds = {
            id(variable): Bound(-10, 10),
        }
        variables = [variable]
        for lb, ub in [(0, 1), (-1, 3)]:
            v = dex.Variable('x', lb, ub)
            variables.append(v)
            bounds[id(v)] = Bound(lb, ub)

        linear = dex.LinearExpression(coefficients=[-0.5, 2.0, -1.2], children=variables)
        constraint = dex.Constraint('c0', lower_bound=0, upper_bound=20, children=[linear])
        bounds[id(constraint)] = Bound(0, 20)
        new_bound = tighten_variable_in_linear_component(linear, [], variable, constraint, bounds)
        assert new_bound == Bound(-10, 6.4)

    def test_tighten_variables_with_nonlinear_component(self):
        variable = dex.Variable('x0', -100, 100)
        bounds = {
            id(variable): Bound(-100, 100),
        }
        variables = [variable]
        for lb, ub in [(0, 1), (-1, 3)]:
            v = dex.Variable('x', lb, ub)
            variables.append(v)
            bounds[id(v)] = Bound(lb, ub)

        linear = dex.LinearExpression(coefficients=[-0.5, 2.0, -1.2], children=variables)

        p0 = PlaceholderExpression()
        bounds[id(p0)] = Bound(-10, 1)
        p1 = PlaceholderExpression()
        bounds[id(p1)] = Bound(2, None)

        root = dex.SumExpression([p0, linear, p1])
        constraint = dex.Constraint('c0', lower_bound=0, upper_bound=20, children=[root])
        bounds[id(constraint)] = Bound(0, 20)

        new_bound = tighten_variable_in_linear_component(linear, [p0, p1], variable, constraint, bounds)
        assert new_bound == Bound(-63.2, 100.0)

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
import suspect.dag.expressions as dex
from tests.conftest import PlaceholderExpression


class TestConstraintLinearComponent(object):
    def test_is_only_child(self):
        root = dex.LinearExpression(coefficients=[1.0], children=[PlaceholderExpression()])
        constr = dex.Constraint('c0', 0, 1, [root])
        assert constr.linear_component() == root

    def test_no_linear_expression(self):
        root = dex.SumExpression(children=[PlaceholderExpression()])
        constr = dex.Constraint('c0', 0, 1, [root])
        assert constr.linear_component() == None

    def test_as_child_of_sum(self):
        linear = dex.LinearExpression(coefficients=[1.0], children=[PlaceholderExpression()])
        root = dex.SumExpression(children=[PlaceholderExpression(), linear, PlaceholderExpression()])
        constr = dex.Constraint('c0', 0, 1, [root])
        assert constr.linear_component() == linear

    def test_malformed_expression(self):
        linear_0 = dex.LinearExpression(coefficients=[1.0], children=[PlaceholderExpression()])
        linear_1 = dex.LinearExpression(coefficients=[1.0], children=[PlaceholderExpression()])
        root = dex.SumExpression(children=[PlaceholderExpression(), linear_0, linear_1])
        constr = dex.Constraint('c0', 0, 1, [root])
        with pytest.raises(AssertionError):
            constr.linear_component()


class TestConstraintNonlinearComponent(object):
    def test_is_only_child(self):
        root = dex.SinExpression([PlaceholderExpression()])
        constr = dex.Constraint('c0', 0, 1, [root])
        assert constr.nonlinear_component() == [root]

    def test_no_nonlinear_expression(self):
        root = dex.LinearExpression(coefficients=[1.0], children=[PlaceholderExpression()])
        constr = dex.Constraint('c0', 0, 1, [root])
        assert constr.nonlinear_component() == None

    def test_as_child_of_sum(self):
        linear = dex.LinearExpression(coefficients=[1.0], children=[PlaceholderExpression()])
        nonlinear_0 = dex.ExpExpression([PlaceholderExpression()])
        nonlinear_1 = dex.LogExpression([PlaceholderExpression()])
        root = dex.SumExpression(children=[nonlinear_0, linear, nonlinear_1])
        constr = dex.Constraint('c0', 0, 1, [root])
        assert constr.nonlinear_component() == [nonlinear_0, nonlinear_1]

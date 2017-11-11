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
import pyomo.environ as aml
from convexity_detection.bounds import Bound
from convexity_detection.expr_visitor import (
    LinearExpression,
    SumExpression,
)
from convexity_detection.tightening import *
from convexity_detection.math import almosteq
from fixtures import model


def test_tighten_variable_bounds_missing_var(model):
    model.test = aml.Constraint(
        rule=lambda m: -10 <= sum(m.z[i] for i in m.I) <= 10
    )

    # Notice it's model.y not model.z!
    bound = tighten_variable_bounds(model.test, model.y[1, 1])
    expected_bound = Bound(0, 1)
    assert expected_bound == bound


def test_tighten_variable_bounds(model):
    model.test = aml.Constraint(
        rule=lambda m: -10 <= sum(2*m.z[i] for i in m.I) <= 10
    )

    bound = tighten_variable_bounds(model.test, model.z[1])
    assert bound == Bound(0.0, 5.0)


def test_remove_var_from_linear_expr(model):
    test1 = sum(model.x[i] for i in model.I)
    test1_no_x0, coef = remove_var_from_linear_expr(test1, model.x[0])
    assert len(test1_no_x0._args) == len(model.I) - 1
    assert almosteq(coef, 1.0)


def test_inequality_bounds(model):
    expr_l = sum(model.x[i] for i in model.I) >= 10.0
    assert Bound(10.0, None) == inequality_bounds(expr_l)

    expr_u = sum(model.x[i] for i in model.I) <= 10.0
    assert Bound(None, 10.0) == inequality_bounds(expr_u)

    expr_lu = -10 <= sum(model.x[i] for i in model.I) <= 10.0
    assert Bound(-10, 10) == inequality_bounds(expr_lu)


def test_linear_nonlinear_components(model):
    # linear component only
    def _test1(m):
        return 0 <= (sum((i+1)*m.x[i] for i in m.I) +
                     sum(m.y[i, j] for i in m.I for j in m.J)) <= 10
    model.test1 = aml.Constraint(rule=_test1)
    linear1, nonlinear1 = linear_nonlinear_components(model.test1.expr)
    assert nonlinear1 is None
    assert isinstance(linear1, LinearExpression)
    assert len(linear1._args) == (len(model.I) + len(model.I) * len(model.J))

    # nonlinear component only
    def _test2(m):
        return sum(m.x[i] * m.x[i-1] for i in m.I[1:]) == 0
    model.test2 = aml.Constraint(rule=_test2)
    linear2, nonlinear2 = linear_nonlinear_components(model.test2.expr)
    assert linear2 is None
    assert isinstance(nonlinear2, SumExpression)
    assert len(nonlinear2._args) == len(model.I[1:])

    # mixe linear nonlinear
    def _test3(m):
        sum1 = sum(i * m.x[i] for i in m.I)
        nl = sum(m.x[i] * m.y[i, j] for i in m.I for j in m.J)
        sum2 = sum(m.y[i, 0] for i in m.I)
        sum3 = sum(m.x[i] for i in m.I)
        return sum1 + nl + sum2 + sum3 >= 0

    model.test3 = aml.Constraint(rule=_test3)
    linear3, nonlinear3 = linear_nonlinear_components(model.test3.expr)
    assert isinstance(nonlinear3, SumExpression)
    assert len(nonlinear3._args) == (len(model.I)*len(model.J))
    # sum1 and sum3 are joined together
    assert len(linear3._args) == 2*len(model.I)

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
from tests.strategies import models
from hypothesis import given, reproduce_failure
import pyomo.environ as aml
from suspect.pyomo.expressions import (
    nonpyomo_leaf_types,
    NumericConstant,
)
import numpy as np
from suspect.dag import ProblemDag
from suspect.pyomo.convert import ComponentFactory, dag_from_pyomo_model
from suspect.pyomo.util import model_variables, model_constraints, model_objectives
from suspect.math.arbitrary_precision import inf


def expression_equal(expected, actual):
    if isinstance(actual, NumericConstant):
        assert isinstance(expected, float)
        return np.isclose(expected, actual.value)
    if type(expected).__name__ != type(actual).__name__:
        return False
    if type(expected) in nonpyomo_leaf_types:
        return expected == actual
    if expected.is_variable_type():
        return actual.is_variable_type()
    if len(expected.args) != len(actual.args):
        return False
    for expected_arg, actual_arg in zip(expected.args, actual.args):
        if not expression_equal(expected_arg, actual_arg):
            return False
    return True


class TestConvertVariable(object):
    def test_continuous_variables(self):
        m = aml.ConcreteModel()
        # 10 continuous variables in [-inf, inf]
        m.x = aml.Var(range(10))

        dag = ProblemDag()
        factory = ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.variable(omo_var)
            assert new_var.name.startswith('x')
            assert new_var.name == omo_var.name
            assert new_var.lb is None
            assert new_var.ub is None
            assert not new_var.is_integer()
            assert not new_var.is_binary()
            count += 1
        assert count == 10

    def test_integer_variables(self):
        m = aml.ConcreteModel()
        # 5 integer variables in [-10, 5]
        m.y = aml.Var(range(5), bounds=(-10, 5), domain=aml.Integers)

        dag = ProblemDag()
        factory = ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.variable(omo_var)
            assert new_var.name.startswith('y')
            assert new_var.lb == -10
            assert new_var.ub == 5
            assert new_var.is_integer()
            count += 1
        assert count == 5

    def test_binary_variables(self):
        m = aml.ConcreteModel()
        # 10 binary variables
        m.b = aml.Var(range(10), domain=aml.Binary)

        dag = ProblemDag()
        factory = ComponentFactory(dag)
        count = 0
        for omo_var in model_variables(m):
            new_var = factory.variable(omo_var)
            assert new_var.name.startswith('b')
            assert new_var.lb == 0
            assert new_var.ub == 1
            assert new_var.is_binary()
            count += 1
        assert count == 10


@given(models(max_constraints=1))
def test_conversion(model):
    dag = dag_from_pyomo_model(model)

    for obj in dag.objectives.values():
        expected_obj = getattr(model, obj.name)
        expected_expr = expected_obj.expr
        assert expression_equal(expected_expr, obj.body)
        print('Done')

    for cons in dag.constraints.values():
        expected_cons = getattr(model, cons.name)
        expected_expr = expected_cons.body
        assert expression_equal(expected_expr, cons.body)

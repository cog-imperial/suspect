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
from suspect.dag import ProblemDag
import suspect.dag.expressions as dex
from suspect.pyomo.convert import ComponentFactory
from suspect.pyomo.util import model_variables


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
            assert new_var.lower_bound is None
            assert new_var.upper_bound is None
            assert new_var.domain == dex.Domain.REALS
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
            assert new_var.lower_bound == -10
            assert new_var.upper_bound == 5
            assert new_var.domain == dex.Domain.INTEGERS
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
            assert new_var.lower_bound == 0
            assert new_var.upper_bound == 1
            assert new_var.domain == dex.Domain.BINARY
            count += 1
        assert count == 10

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
from suspect.util import *
from tests.fixtures import model


def test_model_variables(model):
    assert 10 + 50 + 10 == len([_ for _ in model_variables(model)])


def test_model_constraints(model):
    pytest.skip('pyomo related')
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 0
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum(m.z[i] for i in m.I) <= 5
    )
    model.cons3 = aml.Constraint(
        model.I,
        rule=lambda m, i: -1 <= m.x[i] <= 1
    )
    model.cons4 = aml.Constraint(
        model.I, model.J,
        rule=lambda m, i, j: -20 <= m.x[i] * m.y[i, j] <= 20,
    )
    assert 1 + 1 + 10 + 50 == len([_ for _ in model_constraints(model)])


def test_model_objectives(model):
    model.obj1 = aml.Objective(expr=sum(model.x[i] for i in model.I))
    model.obj2 = aml.Objective(expr=model.x[0] - model.x[1])
    assert 2 == len([_ for _ in model_objectives(model)])

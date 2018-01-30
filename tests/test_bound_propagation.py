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
from suspect.dag import *
from suspect.bound.propagation import propagate_bounds
import suspect.dag.dot as dot
from suspect.pyomo import dag_from_pyomo_model
import pyomo.environ as aml


def model():
    m = aml.ConcreteModel()
    m.x0 = aml.Var(bounds=(None, None))
    m.x1 = aml.Var(bounds=(0, None))
    m.x2 = aml.Var(bounds=(None, 0))
    m.x3 = aml.Var(bounds=(-10, 10))

    m.c0 = aml.Constraint(expr=2*m.x3 + aml.sqrt(m.x1) >= 0)
    m.c1 = aml.Constraint(expr=m.x0**2 <= 5)
    m.c2 = aml.Constraint(expr=5 <= aml.log(m.x1) + aml.sin(m.x2) <= 100)

    m.obj = aml.Objective(expr=m.x0 + m.x1**2 + m.x2**3 + m.x3**4)
    return dag_from_pyomo_model(m)


def test_propagation():
    dag = model()

    bound_ctx = {}
    propagate_bounds(dag, bound_ctx)

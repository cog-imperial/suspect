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
from suspect.dag import ProblemDag
import suspect.dag.expressions as dex
from suspect.monotonicity.propagation import (
    MonotonicityPropagationVisitor,
    propagate_monotonicity,
)
from suspect.bound.propagation import propagate_bounds
import suspect.dag.dot as dot
from suspect.pyomo import dag_from_pyomo_model
import pyomo.environ as aml


class MockMonotonicityPropagationVisitor(MonotonicityPropagationVisitor):
    def __init__(self, bounds, mono):
        super().__init__(bounds)
        self._mono = mono


def test_variable_is_increasing():
    v = MockMonotonicityPropagationVisitor({}, {})
    var = dex.Variable('x0', None, None)
    v(var)
    assert v.get(var).is_nondecreasing()


def test_constant_is_constant():
    v = MockMonotonicityPropagationVisitor({}, {})
    const = dex.Constant(2.0)
    v(const)
    assert v.get(const).is_constant()


class TestConstraint(object):
    def setup_method(self, _func):
        self.x0 = dex.Variable('x0', None, None)
        self.x1 = dex.NegationExpression([self.x0])
        self.const = dex.Constant(1.0)
        self.v = MockMonotonicityPropagationVisitor({}, {})
        self.v(self.x0)
        self.v(self.x1)
        self.v(self.const)

    def test_lower_upper_bound(self):
        c0 = dex.Constraint('c0', 0, 1, [self.x0])
        self.v(c0)
        assert self.v.get(c0).is_unknown()

        c1 = dex.Constraint('c1', 0, 1, [self.const])
        self.v(c1)
        assert self.v.get(c1).is_constant()

    def test_lower_bound_only(self):
        c0 = dex.Constraint('c0', 0, None, [self.x0])
        self.v(c0)
        assert self.v.get(c0).is_nonincreasing()

        c1 = dex.Constraint('c1', 0, None, [self.const])
        self.v(c1)
        assert self.v.get(c1).is_constant()

        c2 = dex.Constraint('c2', 0, None, [self.x1])
        self.v(c2)
        assert self.v.get(c2).is_nondecreasing()

    def test_upper_bound_only(self):
        c0 = dex.Constraint('c0', None, 1, [self.x0])
        self.v(c0)
        assert self.v.get(c0).is_nondecreasing()

        c1 = dex.Constraint('c1', None, 1, [self.const])
        self.v(c1)
        assert self.v.get(c1).is_constant()

        c2 = dex.Constraint('c2', None, 1, [self.x1])
        self.v(c2)
        assert self.v.get(c2).is_nonincreasing()


class TestObjective(object):
    def setup_method(self, _func):
        self.x0 = dex.Variable('x0', None, None)
        self.x1 = dex.NegationExpression([self.x0])
        self.const = dex.Constant(1.0)
        self.v = MockMonotonicityPropagationVisitor({}, {})
        self.v(self.x0)
        self.v(self.x1)
        self.v(self.const)

    def test_min(self):
        o0 = dex.Objective('o0', dex.Sense.MINIMIZE, [self.x0])
        self.v(o0)
        assert self.v.get(o0).is_nondecreasing()

        o1 = dex.Objective('o1', dex.Sense.MINIMIZE, [self.x1])
        self.v(o1)
        assert self.v.get(o1).is_nonincreasing()

        o2 = dex.Objective('o2', dex.Sense.MINIMIZE, [self.const])
        self.v(o2)
        assert self.v.get(o2).is_constant()

    def test_max(self):
        o0 = dex.Objective('o0', dex.Sense.MAXIMIZE, [self.x0])
        self.v(o0)
        assert self.v.get(o0).is_nonincreasing()

        o1 = dex.Objective('o1', dex.Sense.MAXIMIZE, [self.x1])
        self.v(o1)
        assert self.v.get(o1).is_nondecreasing()

        o2 = dex.Objective('o2', dex.Sense.MAXIMIZE, [self.const])
        self.v(o2)
        assert self.v.get(o2).is_constant()


@pytest.fixture
def product_dag():
    m = aml.ConcreteModel()
    m.x = aml.Var(bounds=(0, None))
    m.y = aml.Var(bounds=(None, 0))

    # positive increasing
    pi = m.x
    # negative increasing
    ni = m.y
    # positive decreasing
    pd = -ni
    # negative decreasing
    nd = -pi

    # mul by positive constant
    m.pi_c = aml.Constraint(expr=pi*2.0 <= 100)
    m.ni_c = aml.Constraint(expr=ni*2.0 <= 100)
    m.pd_c = aml.Constraint(expr=pd*2.0 <= 100)
    m.nd_c = aml.Constraint(expr=nd*2.0 <= 100)

    # mul by negative constant
    m.pi_nc = aml.Constraint(expr=pi*-2.0 <= 100)
    m.ni_nc = aml.Constraint(expr=ni*-2.0 <= 100)
    m.pd_nc = aml.Constraint(expr=pd*-2.0 <= 100)
    m.nd_nc = aml.Constraint(expr=nd*-2.0 <= 100)

    # other cases
    m.pi_pi = aml.Constraint(expr=pi*pi <= 100)
    m.pi_ni = aml.Constraint(expr=pi*ni <= 100)
    m.pi_pd = aml.Constraint(expr=pi*pd <= 100)
    m.pi_nd = aml.Constraint(expr=pi*nd <= 100)

    m.ni_pi = aml.Constraint(expr=ni*pi <= 100)
    m.ni_ni = aml.Constraint(expr=ni*ni <= 100)
    m.ni_pd = aml.Constraint(expr=ni*pd <= 100)
    m.ni_nd = aml.Constraint(expr=ni*nd <= 100)

    m.pd_pi = aml.Constraint(expr=pd*pi <= 100)
    m.pd_ni = aml.Constraint(expr=pd*ni <= 100)
    m.pd_pd = aml.Constraint(expr=pd*pd <= 100)
    m.pd_nd = aml.Constraint(expr=pd*nd <= 100)

    m.nd_pi = aml.Constraint(expr=nd*pi <= 100)
    m.nd_ni = aml.Constraint(expr=nd*ni <= 100)
    m.nd_pd = aml.Constraint(expr=nd*pd <= 100)
    m.nd_nd = aml.Constraint(expr=nd*nd <= 100)

    return dag_from_pyomo_model(m)


def test_product(product_dag):
    bounds = {}
    dag = product_dag
    propagate_bounds(dag, bounds)
    mono = propagate_monotonicity(dag, bounds)

    c = dag.constraints

    assert mono[id(c['pi_c'])].is_nondecreasing()
    assert mono[id(c['ni_c'])].is_nondecreasing()
    assert mono[id(c['pd_c'])].is_nonincreasing()
    assert mono[id(c['nd_c'])].is_nonincreasing()

    assert mono[id(c['pi_nc'])].is_nonincreasing()
    assert mono[id(c['ni_nc'])].is_nonincreasing()
    assert mono[id(c['pd_nc'])].is_nondecreasing()
    assert mono[id(c['nd_nc'])].is_nondecreasing()

    assert mono[id(c['pi_pi'])].is_nondecreasing()
    assert mono[id(c['pi_ni'])].is_unknown()
    assert mono[id(c['pi_pd'])].is_unknown()
    assert mono[id(c['pi_nd'])].is_nonincreasing()

    assert mono[id(c['ni_pi'])].is_unknown()
    assert mono[id(c['ni_ni'])].is_nonincreasing()
    assert mono[id(c['ni_pd'])].is_nondecreasing()
    assert mono[id(c['ni_nd'])].is_unknown()

    assert mono[id(c['pd_pi'])].is_unknown()
    assert mono[id(c['pd_ni'])].is_nondecreasing()
    assert mono[id(c['pd_pd'])].is_nonincreasing()
    assert mono[id(c['pd_nd'])].is_unknown()

    assert mono[id(c['nd_pi'])].is_nonincreasing()
    assert mono[id(c['nd_ni'])].is_unknown()
    assert mono[id(c['nd_pd'])].is_unknown()
    assert mono[id(c['nd_nd'])].is_nondecreasing()

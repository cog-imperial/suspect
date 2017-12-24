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
import hypothesis.strategies as st
from suspect.dag import *


@pytest.fixture
def model():
    m = aml.ConcreteModel()
    m.I = range(10)
    m.J = range(5)
    m.x = aml.Var(m.I)
    m.y = aml.Var(m.I, m.J, domain=aml.NonNegativeReals, bounds=(0, 1))
    m.z = aml.Var(m.I, bounds=(0, 10))
    return m


@st.composite
def variables(draw):
    domain = draw(st.sampled_from(
        Domain.REALS,
        Domain.INTEGERS,
        Domain.BINARY,
        ))
    lower_bound = draw(st.one_of(st.none(), st.floats()))
    upper_bound = draw(st.one_of(st.none(), st.floats(min_value=lower_bound)))
    return Variable(lower_bound, upper_bound, domain)

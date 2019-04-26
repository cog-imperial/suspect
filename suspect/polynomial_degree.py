# Copyright 2018 Francesco Ceccon
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

from pyomo.core.kernel.component_map import ComponentMap
from suspect.dag.iterator import DagForwardIterator
from suspect.polynomial import PolynomialDegreeVisitor


def polynomial_degree(dag):
    """Compute polynomial degree of expressions in dag."""
    iterator = DagForwardIterator()
    polynomial = ComponentMap()
    iterator.iterate(dag, PolynomialDegreeVisitor(), polynomial)
    return polynomial

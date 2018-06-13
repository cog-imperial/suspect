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

"""Directed Acyclic Graph representation of an optimization problem."""

from suspect.interfaces import Problem
from suspect.dag.vertices_list import VerticesList


class ProblemDag(Problem):
    r"""The optimization problem represented as Directed Acyclic Graph (DAG).

    The vertices in the DAG are sorted by depth, defined as

    .. math::

        d(v) = \begin{cases}
          0 & \text{if } v \text{ is a variable or constant}\\
          \max\{d(u) | u \in c(v)\} & \text{otherwise}
        \end{cases}

    where :math:`c(v)` are the children of vertex :math:`v`.


    Attributes
    ----------
    name : str
        the problem name.
    variables : dict
        the problem variables.
    constraints : dict
        the problem constraints.
    objectives : dict
        the problem objectives.
    vertices : iter
        an iterator over the vertices sorted by :math:`d(v)`.
    """
    def __init__(self, name=None):
        self.name = name
        # The DAG vertices sorted by depth
        self._vertices = VerticesList()
        # Vertices that have no children
        self._sources = []
        # Vertices that have no parent
        self._sinks = []
        # A pointer to vertices that are variables
        self.variables = {}
        # A pointer to vertices that are constraints
        self.constraints = {}
        # A pointer to vertices that are objectives
        self.objectives = {}

    @property
    def vertices(self):
        """Return an iterator over the problem vertices."""
        return iter(self._vertices)

    def add_vertex(self, vertex):
        """Add a vertex to the DAG.

        Parameters
        ----------
        vertex : Expression
           the vertex to add.
        """
        self._vertices.append(vertex)
        if vertex.is_source:
            self._sources.append(vertex)

        if vertex.is_sink:
            self._sinks.append(vertex)

    def _add_named(self, expr, collection):
        self.add_vertex(expr)
        collection[expr.name] = expr

    def add_variable(self, var):
        """Add a variable to the DAG.

        Parameters
        ----------
        var : Variable
            the variable.
        """
        self._add_named(var, self.variables)

    def add_constraint(self, cons):
        """Add a constraint to the DAG.

        Parameters
        ----------
        cons : Constraint
           the constraint.
        """
        self._add_named(cons, self.constraints)

    def add_objective(self, obj):
        """Add an objective to the DAG.

        Parameters
        ----------
        obj : Objective
           the objective.
        """
        self._add_named(obj, self.objectives)

    def stats(self):
        """Return statistics about the DAG.

        Returns
        -------
        dict
            A dictionary containing information about the DAG:

             * Number of vertices
             * Maximum depth
             * Number of variables
             * Number of constraints
             * Number of objectives
        """
        return {
            'num_vertices': len(self.vertices),
            'num_variables': len(self.variables),
            'num_constraints': len(self.constraints),
            'num_objectives': len(self.objectives),
        }

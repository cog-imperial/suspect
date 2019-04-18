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

from pyomo.core.expr.numeric_expr import nonpyomo_leaf_types
from suspect.interfaces import Problem
from suspect.dag.expressions import Constraint, Objective
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
        self._depths = {}
        # Keep a list of vertices that depend on a vertex
        self._parents = {}
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
        depth = self._vertex_depth(vertex)
        self._depths[id(vertex)] = depth
        self._vertices.append(vertex, depth)
        self._add_parents(vertex)
        if _is_source_type(vertex):
            self._sources.append(vertex)

        if _is_sink_type(vertex):
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

    def depth(self, vertex):
        if type(vertex) in nonpyomo_leaf_types:
            return 2
        if vertex.is_variable_type():
            return 1
        if vertex.is_constant():
            return 2
        return self._depths[id(vertex)]

    def _vertex_depth(self, vertex):
        assert id(vertex) not in self._depths
        if vertex.is_variable_type():
            return 1
        if vertex.is_constant():
            return 2
        depth = 0
        for arg in vertex.args:
            arg_depth = self.depth(arg)
            if arg_depth >= depth:
                depth = arg_depth + 1
        return depth

    def parents(self, vertex):
        if isinstance(vertex, (Constraint, Objective)):
            return []
        return self._parents[id(vertex)].values()

    def _add_parents(self, vertex):
        # For each of the args of vertex, add vertex as parent
        # Parents are stored in a dict so that we have both id and node
        if type(vertex) in nonpyomo_leaf_types:
            return

        if not vertex.is_expression_type():
            return

        for arg in vertex.args:
            if id(arg) not in self._parents:
                self._parents[id(arg)] = {}
            parents = self._parents[id(arg)]
            parents[id(vertex)] = vertex


def _is_source_type(node):
    is_nonpyomo_leaf = node.__class__ in nonpyomo_leaf_types
    is_variable = node.is_variable_type()
    is_constant = node.is_constant()
    return is_nonpyomo_leaf or is_variable or is_constant


def _is_sink_type(node):
    return isinstance(node, (Constraint, Objective))

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

"""Iterators over vertices of ProblemDag."""
import abc
from suspect.pyomo.expressions import nonpyomo_leaf_types
from suspect.interfaces import ForwardIterator, BackwardIterator
from suspect.dag.dag import ProblemDag
from suspect.dag.expressions import Expression
from suspect.dag.vertices_list import VerticesList


class ForwardDagIterator(object):
    def iterate(self, problem, visitor, *args, starting_vertices=None):
        sources = set(id(s) for s in problem._sources)
        if starting_vertices is None:
            starting_vertices = sources
        else:
            starting_vertices = set(id(v) for v in starting_vertices) | sources

        results = {}
        changed = set()
        for vertex in problem.vertices:
            has_changed = visitor.visit(vertex, *args)
            if has_changed:
                changed.add(id(vertex))
        return list(changed)


class _DagIterator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _get_next_vertices(self, problem, expr):
        pass

    @abc.abstractproperty
    def _reverse(self):
        pass

    def _iterate(self, problem, visitor, starting_vertices, *args):
        changed_vertices = []
        vertices = VerticesList(reverse=self._reverse)
        for vertex in starting_vertices:
            vertices.append(vertex, problem.depth(vertex))
        seen = set()
        while vertices:
            curr_vertex = vertices.pop()
            if id(curr_vertex) in seen:
                continue
            has_changes = visitor.visit(curr_vertex, *args)
            seen.add(id(curr_vertex))

            if has_changes:
                changed_vertices.append(curr_vertex)
                for vertex in self._get_next_vertices(problem, curr_vertex):
                    if id(vertex) not in seen:
                        vertices.append(vertex, problem.depth(vertex))

        return changed_vertices


class DagForwardIterator(_DagIterator, ForwardIterator):
    """Forward iterator over suspect.dag.ProblemDag vertices."""
    def _get_next_vertices(self, problem, expr):
        if type(expr) in nonpyomo_leaf_types:
            return []
        return problem.parents(expr)

    @property
    def _reverse(self):
        return False

    def iterate(self, problem, visitor, *args, **kwargs): # pylint: disable=missing-docstring
        starting_vertices = kwargs.pop('starting_vertices', None)
        # pylint: disable=protected-access
        if starting_vertices is None:
            starting_vertices = problem._sources
        else:
            starting_vertices = problem._sources + starting_vertices

        return super()._iterate(
            problem,
            visitor,
            starting_vertices,
            *args,
        )


class DagBackwardIterator(_DagIterator, BackwardIterator):
    """Backward iterator over suspect.dag.ProblemDag vertices."""
    def _get_next_vertices(self, _problem, expr):
        if not expr.is_expression_type():
            return []
        return expr.args

    @property
    def _reverse(self):
        return True

    def iterate(self, problem, visitor, *args, **kwargs): # pylint: disable=missing-docstring
        starting_vertices = kwargs.pop('starting_vertices', None)
        # pylint: disable=protected-access
        if starting_vertices is None:
            starting_vertices = problem._sinks
        else:
            starting_vertices = problem._sinks + starting_vertices

        return super()._iterate(
            problem,
            visitor,
            starting_vertices,
            *args,
        )

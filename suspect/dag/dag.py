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

import bisect


def _reverse_bisect_right(arr, x):
    """Like bisect.bisect_right, but insert in a descending list"""
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class VerticesList(object):
    """A list of vertices sorted by their depth."""
    def __init__(self, vertices=None, reverse=False):
        if vertices is None:
            vertices = []
        else:
            vertices = sorted(vertices, key=lambda v: v.depth, reverse=reverse)

        self._vertices = vertices
        self._vertices_depth = [v.depth for v in self._vertices]
        self._reverse = reverse

        if self._reverse:
            self._find_insertion_idx = _reverse_bisect_right
        else:
            self._find_insertion_idx = bisect.bisect_right

    def append(self, vertex):
        """Append vertex to the list, keeping the vertices sorted by depth"""
        depth = vertex.depth
        insertion_idx = self._find_insertion_idx(self._vertices_depth, depth)
        self._vertices.insert(insertion_idx, vertex)
        self._vertices_depth.insert(insertion_idx, depth)

    def pop(self):
        """Pop an element from the front of the list"""
        self._vertices_depth.pop(0)
        return self._vertices.pop(0)

    def __iter__(self):
        return iter(self._vertices)

    def __len__(self):
        return len(self._vertices)


class ProblemDag(object):
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
        return iter(self._vertices)

    def forward_visit(self, cb, ctx, starting_vertices=None):
        if starting_vertices is None:
            starting_vertices = self._sources
        return self._visit(
            cb,
            ctx,
            starting_vertices,
            get_next_vertices=lambda c: c.parents,
            reverse=False,
        )

    def backward_visit(self, cb, ctx, starting_vertices=None):
        if starting_vertices is None:
            starting_vertices = self._sinks
        return self._visit(
            cb,
            ctx,
            starting_vertices,
            get_next_vertices=lambda c: [c],
            reverse=True,
        )

    def _visit(self, cb, ctx, starting_vertices, get_next_vertices, reverse):
        changed_vertices = []
        vertices = VerticesList(starting_vertices, reverse=reverse)
        seen = set()
        while len(vertices) > 0:
            curr_vertex = vertices.pop()
            changes = cb(curr_vertex, ctx)
            seen.add(id(curr_vertex))

            if changes is not None:
                for v in changes:
                    changed_vertices.append(v)
                    for next_vertex in get_next_vertices(v):
                        if id(next_vertex) not in seen:
                            vertices.append(next_vertex)
                            seen.add(id(next_vertex))

        return changed_vertices

    def add_vertex(self, vertex):
        self._vertices.append(vertex)
        if vertex.is_source:
            self._sources.append(vertex)

        if vertex.is_sink:
            self._sinks.append(vertex)

    def _add_named(self, expr, collection):
        self.add_vertex(expr)
        collection[expr.name] = expr

    def add_variable(self, var):
        self._add_named(var, self.variables)

    def add_constraint(self, cons):
        self._add_named(cons, self.constraints)

    def add_objective(self, obj):
        self._add_named(obj, self.objectives)

    def stats(self):
        return {
            'num_vertices': len(self.vertices),
            'max_depth': max(self._vertices_depth),
            'num_variables': len(self.variables),
            'num_constraints': len(self.constraints),
            'num_objectives': len(self.objectives),
        }

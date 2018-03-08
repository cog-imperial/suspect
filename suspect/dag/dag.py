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


class ProblemDag(object):
    def __init__(self, name=None):
        self.name = name
        # The DAG vertices sorted by depth
        self.vertices = []
        # Precompute depth of vertices to use in bisect
        self._vertices_depth = []
        # A pointer to vertices that are variables
        self.variables = {}
        # A pointer to vertices that are constraints
        self.constraints = {}
        # A pointer to vertices that are objectives
        self.objectives = {}

    def forward_visit(self, cb, ctx):
        for v in self.vertices:
            cb(v, ctx)

    def bacward_visit(self, cb, ctx):
        for v in reversed(self.vertices):
            cb(v, ctx)

    def add_vertex(self, vertex):
        depth = vertex.depth
        insertion_idx = bisect.bisect_left(self._vertices_depth, depth)
        self.vertices.insert(insertion_idx, vertex)
        self._vertices_depth.insert(insertion_idx, depth)

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

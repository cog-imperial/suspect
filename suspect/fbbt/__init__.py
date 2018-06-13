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

"""Feasibility Based Bound Tightening module."""
from typing import Generic
from suspect.interfaces import ForwardIterator, BackwardIterator, Problem, V, C
from .initialization import BoundsInitializationVisitor
from .propagation import BoundsPropagationVisitor
from .tightening import BoundsTighteningVisitor
import suspect.dag.dot as dot


class FBBTStopCriterion(object):
    def __init__(self, max_iter=10):
        self._max_iter = max_iter
        self._iter = 0

    def should_stop(self):
        return self._iter >= self._max_iter

    def iteration_end(self):
        self._iter += 1

    def intercept_changes(self, visitor):
        if not hasattr(visitor, 'handle_result'):
            raise RuntimeError('expected decorated visitor to have handle_result method')
        original_visitor_handle = visitor.handle_result
        def _wrapper(expr, new_bound, ctx):
            return original_visitor_handle(expr, new_bound, ctx)
        visitor.handle_result = _wrapper
        return visitor


class BoundsTightener(Generic[V, C]):
    def __init__(self, forward_iterator: ForwardIterator[V, C],
                 backward_iterator: BackwardIterator[V, C],
                 stop_criterion: FBBTStopCriterion):
        self._forward_iterator = forward_iterator
        self._backward_iterator = backward_iterator
        self._stop_criterion = stop_criterion

    def tighten(self, problem: Problem[V], ctx: C):
        self._forward_iterator.iterate(problem, BoundsInitializationVisitor(), ctx)
        dot.dump(problem, open('/tmp/dag-init.dot', 'w'), ctx)
        prop_visitor = self._stop_criterion.intercept_changes(BoundsPropagationVisitor())
        tigh_visitor = self._stop_criterion.intercept_changes(BoundsTighteningVisitor())
        changes_tigh = None
        changes_prop = None
        while not self._stop_criterion.should_stop():
            changes_prop = self._forward_iterator.iterate(
                problem, prop_visitor, ctx, # starting_vertices=changes_tigh
            )
            changes_tigh = self._backward_iterator.iterate(
                problem, tigh_visitor, ctx, # starting_vertices=changes_prop
            )
            self._stop_criterion.iteration_end()
        dot.dump(problem, open('/tmp/dag-finish.dot', 'w'), ctx)

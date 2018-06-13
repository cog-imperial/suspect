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

"""Class to run bound tightener or a problem."""
from suspect.fbbt.initialization import BoundsInitializationVisitor
from suspect.fbbt.propagation import BoundsPropagationVisitor
from suspect.fbbt.tightening import BoundsTighteningVisitor


class FBBTStopCriterion(object):
    """Stop criterion for FBBT."""
    def __init__(self, max_iter=10):
        self._max_iter = max_iter
        self._iter = 0

    def should_stop(self):
        """Predicate to check if FBBT should stop."""
        return self._iter >= self._max_iter

    def iteration_end(self):
        """Callback called at the end of each FBBT iteration."""
        self._iter += 1

    def intercept_changes(self, visitor):
        if not hasattr(visitor, 'handle_result'):
            raise RuntimeError('expected decorated visitor to have handle_result method')
        original_visitor_handle = visitor.handle_result
        def _wrapper(expr, new_bound, ctx):
            return original_visitor_handle(expr, new_bound, ctx)
        visitor.handle_result = _wrapper
        return visitor


class BoundsTightener(object):
    """Configure and run FBBT on a problem.

    Parameters
    ----------
    forward_iterator:
       forward iterator over vertices of the problem
    backward_iterator:
       backward iterator over vertices of the problem
    stop_criterion:
       criterion used to stop iteration
    """
    def __init__(self, forward_iterator, backward_iterator, stop_criterion):
        self._forward_iterator = forward_iterator
        self._backward_iterator = backward_iterator
        self._stop_criterion = stop_criterion

    def tighten(self, problem, ctx):
        """Tighten bounds of ``problem`` storing them in ``ctx``."""
        self._forward_iterator.iterate(problem, BoundsInitializationVisitor(), ctx)
        prop_visitor = self._stop_criterion.intercept_changes(BoundsPropagationVisitor())
        tigh_visitor = self._stop_criterion.intercept_changes(BoundsTighteningVisitor())
        changes_tigh = None
        changes_prop = None
        while not self._stop_criterion.should_stop():
            changes_prop = self._forward_iterator.iterate(
                problem, prop_visitor, ctx, starting_vertices=changes_tigh
            )
            changes_tigh = self._backward_iterator.iterate(
                problem, tigh_visitor, ctx, starting_vertices=changes_prop
            )
            self._stop_criterion.iteration_end()

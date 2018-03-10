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

from suspect.monotonicity import MonotonicityPropagationVisitor
from suspect.convexity import ConvexityPropagationVisitor
import pkg_resources


def monotonicity_detection_entry_points():
    return pkg_resources.iter_entry_points('suspect.monotonicity_detection')


def convexity_detection_entry_points():
    return pkg_resources.iter_entry_points('suspect.convexity_detection')


class SpecialStructurePropagationVisitor(object):
    def __init__(self):
        self._mono_visitors = [MonotonicityPropagationVisitor()]
        for entry_point in monotonicity_detection_entry_points():
            cls = entry_point.load()
            self._mono_visitors.append(cls())

        self._cvx_visitors = [ConvexityPropagationVisitor()]
        for entry_point in convexity_detection_entry_points():
            cls = entry_point.load()
            self._cvx_visitors.append(cls())

    def __call__(self, expr, ctx):
        for mono_visitor in self._mono_visitors:
            mono_unknown = mono_visitor(expr, ctx)
            if not mono_unknown:
                break

        for cvx_visitor in self._cvx_visitors:
            cvx_unknown = cvx_visitor(expr, ctx)
            if not cvx_unknown:
                break


def propagate_special_structure(problem, ctx):
    """Propagate special structure.

    Arguments
    ---------
    problem: ProblemDag
        the problem.
    ctx: SpecialStructurePropagationContext
      the context containing the bounds

    Returns
    -------
    monotonicity: dict-like
        monotonicity information for the problem.
    convexity: dict-like
        convexity information for the problem.
    """
    visitor = SpecialStructurePropagationVisitor()
    problem.forward_visit(visitor, ctx)
    return ctx.monotonicity, ctx.convexity

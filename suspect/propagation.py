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


class ConvexityMonotonicityPropagationVisitor(object):
    def __init__(self):
        self._mono = MonotonicityPropagationVisitor()
        self._cvx = ConvexityPropagationVisitor()

    def __call__(self, expr, ctx):
        self._mono(expr, ctx)
        self._cvx(expr, ctx)


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
    visitor = ConvexityMonotonicityPropagationVisitor()
    problem.forward_visit(visitor, ctx)
    return ctx.monotonicity, ctx.convexity

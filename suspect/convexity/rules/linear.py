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

"""Convexity detection rules for linear and sum expressions."""
from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule


class LinearRule(ConvexityRule):
    """Return convexity of linear expression."""
    def apply(self, expr, convexity, mono, bounds):
        cvxs = [
            _adjust_convexity(convexity[child], expr.coefficient(child))
            for child in expr.args
        ]
        return _combine_convexities(cvxs)


class SumRule(ConvexityRule):
    """Return convexity of sum expression."""
    def apply(self, expr, convexity, _mono, _bounds):
        cvxs = [convexity[child] for child in expr.args]
        return _combine_convexities(cvxs)


class ExpressionRule(ConvexityRule):
    """Return convexity of a named Expression"""
    def apply(self, expr, convexity, monotonicity, bounds):
        cvx = convexity[expr.expr]
        return cvx


def _adjust_convexity(cvx, coef):
    if cvx.is_unknown() or cvx.is_linear():
        return cvx

    if coef > 0:
        return cvx
    elif coef < 0:
        return cvx.negate()

    # if coef == 0, it's a constant with value 0.0
    return Convexity.Linear


def _combine_convexities(cvxs):
    all_linear = all([c.is_linear() for c in cvxs])
    all_convex = all([c.is_convex() for c in cvxs])
    all_concave = all([c.is_concave() for c in cvxs])

    if all_linear:
        return Convexity.Linear
    elif all_convex:
        return Convexity.Convex
    elif all_concave:
        return Convexity.Concave

    return Convexity.Unknown

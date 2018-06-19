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
from suspect.expression import ExpressionType
from suspect.interfaces import Rule


class LinearRule(Rule):
    """Return convexity of linear expression."""
    root_expr = ExpressionType.Linear

    def apply(self, expr, ctx):
        cvxs = [
            _adjust_convexity(ctx.convexity(child), coef)
            for child, coef in zip(expr.children, expr.coefficients)
        ]
        return _combine_convexities(cvxs)


class SumRule(Rule):
    """Return convexity of sum expression."""
    root_expr = ExpressionType.Sum

    def apply(self, expr, ctx):
        cvxs = [ctx.convexity(child) for child in expr.children]
        return _combine_convexities(cvxs)


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

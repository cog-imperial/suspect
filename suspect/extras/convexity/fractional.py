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

"""Convexity detector for fractional expressions."""
from suspect.ext import ConvexityDetector
from suspect.convexity import Convexity
from suspect.interfaces import Rule
from suspect.expression import ExpressionType
from suspect.math import almosteq # pylint: disable=no-name-in-module


class FractionalConvexityDetector(ConvexityDetector):
    """Convexity detector for fractional expressions of the type:

        g(x) = (a1*x + b1) / (a2*x + b2)

    the convexity is derived from the second derivative of the expression:

        ddg/ddx = -2*a2*(a1*b2 - a2*b1)/(a1*x + b2)^3
    """
    def __init__(self, _problem):
        super().__init__()
        self._rule = FractionalRule()

    def visit_expression(self, expr, convexity, mono, bounds):
        return False, None


class FractionalRule(Rule):
    """Convexity detector for fractional expressions."""
    linear_types = (
        ExpressionType.Constant,
        ExpressionType.Variable,
        ExpressionType.Linear,
    )

    def apply(self, expr, ctx):
        num, den = expr.children
        if not self._valid_expression(num, den):
            return None

        a1, x, b1 = self._linear_components(num)
        a2, y, b2 = self._linear_components(den)

        if x is not None and y is not None:
            if x is not y:
                return None

        if x is not None:
            x_bound = ctx.bounds(x)
        elif y is not None:
            x_bound = ctx.bounds(y)
        else:
            raise RuntimeError('no variable in numerator or denonimator')

        dd_num = -2*a2*(a1*b2 - a2*b1)
        dd_den = a2*x_bound + b2

        if almosteq(dd_num, 0):
            return Convexity.Linear

        dd_bound = dd_den * (1/dd_num)

        if dd_bound.is_nonpositive():
            return Convexity.Concave
        elif dd_bound.is_nonnegative():
            return Convexity.Convex
        # inconclusive
        return None

    def _valid_expression(self, num, den):
        num_type = num.expression_type
        den_type = den.expression_type
        if num_type not in self.linear_types:
            return False
        if den_type not in self.linear_types:
            return False
        if num_type == ExpressionType.Linear:
            if len(num.children) != 1:
                return False
        if den_type == ExpressionType.Linear:
            if len(den.children) != 1:
                return False
        return True

    def _linear_components(self, expr):
        expr_type = expr.expression_type
        if expr_type == ExpressionType.Linear:
            assert len(expr.children) == 1
            x = expr.children[0]
            a = expr.coefficient(x)
            b = expr.constant_term
            return a, x, b
        if expr_type == ExpressionType.Constant:
            return 0.0, None, expr.value
        if expr_type == ExpressionType.Variable:
            return 1.0, expr, 0.0
        raise RuntimeError('invalid expression type in fractional convexity detector')

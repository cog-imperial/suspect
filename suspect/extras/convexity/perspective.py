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
import numpy as np
from suspect.ext import ConvexityDetector
from suspect.convexity import Convexity
from suspect.interfaces import Rule
from suspect.expression import ExpressionType, UnaryFunctionType


class PerspectiveFunctionConvexityDetector(ConvexityDetector):
    """Convexity detector for perspective functions."""
    def __init__(self, _problem):
        super().__init__()

    def visit_expression(self, expr, convexity, mono, bounds):
        return False, None
        return {
            ExpressionType.Product: PerspectiveRule(),
        }


class PerspectiveRule(Rule):
    """Convexity detector for perspective functions."""
    root_expr = ExpressionType.Product

    @property
    def _convex_concave_functions(self):
        return self._convex_functions + self._concave_functions

    @property
    def _convex_functions(self):
        return [UnaryFunctionType.Exp]

    @property
    def _concave_functions(self):
        return [UnaryFunctionType.Sqrt, UnaryFunctionType.Log]

    def apply(self, expr, ctx):
        f, g = expr.children
        f_type = f.expression_type
        g_type = g.expression_type
        if f_type == ExpressionType.Linear and g_type == ExpressionType.Sum:
            return self._detect_syn_convexity_outer(f, g, ctx)
        elif g_type == ExpressionType.Linear and f_type == ExpressionType.Sum:
            return self._detect_syn_convexity_outer(g, f, ctx)
        elif f_type == ExpressionType.Linear and g_type == ExpressionType.UnaryFunction:
            return self._detect_syn_convexity(f, g, ctx)
        elif g_type == ExpressionType.Linear and f_type == ExpressionType.UnaryFunction:
            return self._detect_syn_convexity(g, f, ctx)
        return None

    def _detect_syn_convexity_outer(self, linear_expr, sum_expr, ctx):
        non_div_children = []
        for expr in sum_expr.children:
            if expr.expression_type == ExpressionType.Division:
                cvx_num = ctx.convexity(expr.children[0])
                if not (expr.children[1] is linear_expr and cvx_num.is_linear()):
                    return None
            else:
                non_div_children.append(expr)

        if len(non_div_children) != 1:
            return

        # Drill down to find g
        curr_expr = non_div_children[0]
        sign = 1.0
        while True:
            is_unary_func = curr_expr.expression_type == ExpressionType.UnaryFunction
            if is_unary_func and curr_expr.func_type in self._convex_concave_functions:
                break  # unary function found
            elif curr_expr.expression_type == ExpressionType.Product:
                a, b = curr_expr.children
                a_type = a.expression_type
                b_type = b.expression_type
                if a_type == ExpressionType.Constant and b_type == ExpressionType.UnaryFunction:
                    curr_expr = b
                    sign *= np.sign(a.value)
                elif b_type == ExpressionType.Constant and a_type == ExpressionType.UnaryFunction:
                    curr_expr = a
                    sign *= np.sign(b.value)
                else:
                    return
            elif curr_expr.expression_type == ExpressionType.Negation:
                curr_expr = curr_expr.children[0]
                sign *= -1.0
            else:
                return

        cvx = self._detect_syn_convexity(linear_expr, curr_expr, ctx)

        if cvx is None:
            return None
        if sign > 0:
            return cvx
        return cvx.negate()

    def _detect_syn_convexity(self, linear_expr, unary_expr, _ctx):
        g_func = unary_expr.func_type
        denominator = self._denominator(unary_expr.children[0])

        if denominator is None:
            return None

        if denominator is linear_expr:
            if g_func in self._convex_functions:
                return Convexity.Convex
            elif g_func in self._concave_functions:
                return Convexity.Concave
            else:
                raise RuntimeError('unreachable')
        return None

    def _denominator(self, expr):
        expr_type = expr.expression_type
        if expr_type == ExpressionType.Division:
            return expr.children[1]
        elif expr_type == ExpressionType.Sum:
            divs = []
            for child in expr.children:
                den = self._denominator(child)
                if den is not None:
                    divs.append(den)
            if len(divs) == 1:
                return divs[0]
        return None

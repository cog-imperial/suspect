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

"""Convexity detector for quadratic expressions."""
import numpy as np
from suspect.ext import ConvexityDetector
from suspect.convexity import Convexity
from suspect.interfaces import Rule
from suspect.expression import ExpressionType


class QuadraticFormConvexityDetector(ConvexityDetector):
    """Detect convexity of quadratic expressions xQx."""
    def __init__(self, _problem):
        super().__init__()

    def visit_expression(self, expr, convexity, mono, bounds):
        return False, None
        return {
            ExpressionType.Product: QuadraticRule(),
        }


class QuadraticRule(Rule):
    """Detect convexity of quadratic expressions xQx."""
    root_expr = ExpressionType.Sum

    @property
    def _allowed_children(self):
        return (
            ExpressionType.Product,
            ExpressionType.Negation,
            ExpressionType.Power,
        )

    def apply(self, expr, ctx):
        if not ctx.polynomial(expr).is_quadratic():
            return

        # Check convexity of non quadratic children
        for child in expr.children:
            if not ctx.polynomial(child).is_quadratic():
                if not ctx.convexity(child).is_convex():
                    # we have sum of quadratic + unknown nonlinear
                    # It can't be convex so just stop checking
                    return
            else:
                if child.expression_type not in self._allowed_children:
                    return

        count = 0
        unique_vars = {}
        for (vi, vj, _) in self._variables_of_quadratic(expr, ctx):
            if vi not in unique_vars:
                unique_vars[vi] = count
                count += 1
            if vj not in unique_vars:
                unique_vars[vj] = count
                count += 1

        n = len(unique_vars)
        A = np.zeros((n, n))
        for (vi, vj, c) in self._variables_of_quadratic(expr, ctx):
            i = unique_vars[vi]
            j = unique_vars[vj]
            if i == j:
                A[i, j] = c
            else:
                A[i, j] = c / 2
                A[j, i] = A[i, j]
        eigv = np.linalg.eigvalsh(A)
        if np.all(eigv >= 0):
            return Convexity.Convex
        elif np.all(eigv <= 0):
            return Convexity.Concave

    def _variables_of_quadratic(self, expr, ctx):
        def _both_variables(a, b):
            return (a, b, 1.0)

        def _variable_and_linear(a, b):
            bc = b.children[0]
            return (a, bc, b.coefficient(bc))

        def _const_pow_expression(const, powe):
            a, b = powe.children
            a_type = a.expression_type
            b_type = b.expression_type

            if a_type == ExpressionType.Variable:
                assert b_type == ExpressionType.Constant
                return (a, a, const)
            elif b_type == ExpressionType.Variable:
                assert a_type == ExpressionType.Constant
                return (b, b, const)
            else:
                raise RuntimeError('unreachable')

        def _product_expression(expr):
            a, b = expr.children
            a_type = a.expression_type
            b_type = b.expression_type
            if a_type == ExpressionType.Variable and b_type == ExpressionType.Variable:
                return _both_variables(a, b)
            elif a_type == ExpressionType.Variable and b_type == ExpressionType.Linear:
                return _variable_and_linear(a, b)
            elif b_type == ExpressionType.Variable and a_type == ExpressionType.Linear:
                return _variable_and_linear(b, a)
            elif a_type == ExpressionType.Constant and b_type == ExpressionType.Power:
                return _const_pow_expression(a.value, b)
            elif b_type == ExpressionType.Constant and a_type == ExpressionType.Power:
                return _const_pow_expression(b.value, a)
            else:
                raise RuntimeError('unreachable')

        def _child_variables(child):
            child_type = child.expression_type
            if child_type == ExpressionType.Product:
                return _product_expression(child)
            elif child_type == ExpressionType.Power:
                return _const_pow_expression(1.0, child)
            elif child_type == ExpressionType.Negation:
                expr = child.children[0]
                return _child_variables(expr)
            else:
                raise RuntimeError('unreachable')

        for child in expr.children:
            if not ctx.polynomial[child].is_quadratic():
                continue
            yield _child_variables(child)

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

import numpy as np
import suspect.dag.expressions as dex
from suspect.dag.visitor import ForwardVisitor
from suspect.convexity import Convexity


class RSynConvexityVisitor(ForwardVisitor):
    def register_handlers(self):
        return {
            dex.ProductExpression: self.visit_product,
        }

    def handle_result(self, expr, result, ctx):
        ctx.convexity[expr] = result
        return not result.is_unknown()

    @property
    def convex_concave_functions(self):
        return self.convex_functions + self.concave_functions

    @property
    def convex_functions(self):
        return ['exp']

    @property
    def concave_functions(self):
        return ['sqrt', 'log']

    def visit_product(self, expr, ctx):
        f, g = expr.children
        if isinstance(f, dex.LinearExpression) and isinstance(g, dex.SumExpression):
            return self.detect_syn_convexity(f, g, ctx)
        elif isinstance(g, dex.LinearExpression) and isinstance(f, dex.SumExpression):
            return self.detect_syn_convexity(g, f, ctx)
        elif isinstance(f, dex.LinearExpression) and isinstance(g, dex.UnaryFunctionExpression):
            return self._detect_syn_convexity(f, g, ctx)
        elif isinstance(g, dex.LinearExpression) and isinstance(f, dex.UnaryFunctionExpression):
            return self._detect_syn_convexity(g, f, ctx)

    def detect_syn_convexity(self, linear_expr, sum_expr, ctx):
        non_div_children = []
        for expr in sum_expr.children:
            if isinstance(expr, dex.DivisionExpression):
                cvx_num = ctx.convexity[expr.children[0]]
                if not (expr.children[1] is linear_expr and cvx_num.is_linear()):
                    return
            else:
                non_div_children.append(expr)

        if len(non_div_children) != 1:
            return

        # Drill down to find g
        curr_expr = non_div_children[0]
        sign = 1.0
        while True:
            if isinstance(curr_expr, dex.UnaryFunctionExpression) and \
              curr_expr.func_name in self.convex_concave_functions:
                break  # unary function found
            elif isinstance(curr_expr, dex.ProductExpression):
                a, b = curr_expr.children
                if isinstance(a, dex.Constant) and isinstance(b, dex.UnaryFunctionExpression):
                    curr_expr = b
                    sign *= np.sign(a.value)
                elif isinstance(b, dex.Constant) and isinstance(a, dex.UnaryFunctionExpression):
                    curr_expr = a
                    sign *= np.sign(b.value)
                else:
                    return
            elif isinstance(curr_expr, dex.NegationExpression):
                curr_expr = curr_expr.children[0]
                sign *= -1.0
            else:
                return

        cvx = self._detect_syn_convexity(linear_expr, curr_expr, ctx)

        if cvx is None:
            return
        if sign > 0:
            return cvx
        else:
            return cvx.negate()

    def _detect_syn_convexity(self, linear_expr, unary_expr, ctx):
        g_func = unary_expr.func_name
        denominator = self._denominator(unary_expr.children[0])

        if denominator is None:
            return

        if denominator is linear_expr:
            if g_func in self.convex_functions:
                return Convexity.Convex
            elif g_func in self.concave_functions:
                return Convexity.Concave
            else:
                raise RuntimeError('unreachable')

    def _denominator(self, expr):
        if isinstance(expr, dex.DivisionExpression):
            return expr.children[1]
        elif isinstance(expr, dex.SumExpression):
            divs = []
            for child in expr.children:
                den = self._denominator(child)
                if den is not None:
                    divs.append(den)
            if len(divs) == 1:
                return divs[0]
        return None


class L2NormConvexityVisitor(ForwardVisitor):
    """Detect L_2 Norm convexity in the form

       sqrt(eps + sum_i(x_i**2))
    """
    def register_handlers(self):
        return {
            dex.SqrtExpression: self.visit_sqrt,
        }

    def handle_result(self, expr, result, ctx):
        ctx.convexity[expr] = result
        return not result.is_unknown()

    def visit_sqrt(self, expr, ctx):
        if not isinstance(expr.children[0], dex.SumExpression):
            return
        grandchildren = expr.children[0].children
        if all([self._square_or_constant(c) for c in grandchildren]):
            return Convexity.Convex

    def _square_or_constant(self, expr):
        if isinstance(expr, dex.Constant):
            return True
        if isinstance(expr, dex.PowExpression):
            base, exp = expr.children
            if isinstance(base, dex.Variable) and isinstance(exp, dex.Constant):
                return exp.value == 2
            else:
                return False
        if isinstance(expr, dex.ProductExpression):
            if len(expr.children) != 2:
                return False
            f, g = expr.children
            return f is g and isinstance(f, dex.Variable)
        return False


class QuadraticFormConvexityVisitor(ForwardVisitor):
    def register_handlers(self):
        return {
            dex.SumExpression: self.visit_sum,
        }

    def handle_result(self, expr, result, ctx):
        ctx.convexity[expr] = result
        return not result.is_unknown()

    @property
    def allowed_children(self):
        return (
            dex.ProductExpression,
            dex.NegationExpression,
            dex.PowExpression,
        )

    def visit_sum(self, expr, ctx):
        if not ctx.polynomial[expr].is_quadratic():
            return

        # Check convexity of non quadratic children
        for child in expr.children:
            if not ctx.polynomial[child].is_quadratic():
                if not ctx.convexity[child].is_convex():
                    # we have sum of quadratic + unknown nonlinear
                    # It can't be convex so just stop checking
                    return
            else:
                if not isinstance(child, self.allowed_children):
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
            return (a, b.children[0], b.coefficients[0])

        def _const_pow_expression(const, powe):
            a, b = powe.children
            if isinstance(a, dex.Variable):
                assert isinstance(b, dex.Constant)
                return (a, a, const)
            elif isinstance(b, dex.Variable):
                assert isinstance(a, dex.Constant)
                return (b, b, const)
            else:
                raise RuntimeError('unreachable')

        def _product_expression(expr):
            a, b = expr.children
            if isinstance(a, dex.Variable) and isinstance(b, dex.Variable):
                return _both_variables(a, b)
            elif isinstance(a, dex.Variable) and isinstance(b, dex.LinearExpression):
                return _variable_and_linear(a, b)
            elif isinstance(b, dex.Variable) and isinstance(a, dex.LinearExpression):
                return _variable_and_linear(b, a)
            elif isinstance(a, dex.Constant) and isinstance(b, dex.PowExpression):
                return _const_pow_expression(a.value, b)
            elif isinstance(b, dex.Constant) and isinstance(a, dex.PowExpression):
                return _const_pow_expression(b.value, a)
            else:
                raise RuntimeError('unreachable')

        def _child_variables(child):
            if isinstance(child, dex.ProductExpression):
                return _product_expression(child)
            elif isinstance(child, dex.PowExpression):
                return _const_pow_expression(1.0, child)
            elif isinstance(child, dex.NegationExpression):
                expr = child.children[0]
                return _child_variables(expr)
            else:
                raise RuntimeError('unreachable')

        for child in expr.children:
            if not ctx.polynomial[child].is_quadratic():
                continue
            yield _child_variables(child)

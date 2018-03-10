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

    def visit_product(self, expr, ctx):
        f, g = expr.children
        if isinstance(f, dex.LinearExpression) and isinstance(g, dex.SumExpression):
            return self.detect_syn_convexity(f, g, ctx)
        elif isinstance(g, dex.LinearExpression) and isinstance(f, dex.SumExpression):
            return self.detect_syn_convexity(g, f, ctx)

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

        # Drill down to find log
        curr_expr = non_div_children[0]
        while True:
            if isinstance(curr_expr, dex.UnaryFunctionExpression) and curr_expr.func_name == 'log':
                break
            if isinstance(curr_expr, dex.ProductExpression):
                a, b = curr_expr.children
                if isinstance(a, dex.Constant):
                    curr_expr = b
                else:
                    curr_expr = a
                continue
            curr_expr = curr_expr.children[0]

        if not isinstance(curr_expr.children[0], dex.SumExpression):
            return
        inner_sum = curr_expr.children[0]
        a, b = inner_sum.children
        if isinstance(a, dex.Constant) and isinstance(b, dex.DivisionExpression):
            const = a
            div_expr = b
        elif isinstance(b, dex.Constant) and isinstance(a, dex.DivisionExpression):
            const = b
            div_expr = a
        else:
            return

        if const.value != 1.0:
            return

        num, den = div_expr.children
        if not isinstance(num, dex.Variable):
            return

        if den is linear_expr:
            return Convexity.Convex


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

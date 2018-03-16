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


def polynomial_degree(dag, ctx):
    visitor = PolynomialDegreeVisitor()
    dag.forward_visit(visitor, ctx)
    return ctx.polynomial


class PolynomialDegree(object):
    def __init__(self, degree):
        self.degree = degree

    @classmethod
    def not_polynomial(cls):
        return PolynomialDegree(None)

    def is_polynomial(self):
        return self.degree is not None

    def is_linear(self):
        return self.is_polynomial() and self.degree == 1

    def is_quadratic(self):
        return self.is_polynomial() and self.degree == 2

    def __add__(self, other):
        if self.is_polynomial() and other.is_polynomial():
            return PolynomialDegree(self.degree + other.degree)
        return self.not_polynomial()

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return PolynomialDegree(self.degree * other)
        return PolynomialDegree.not_polynomial()

    def __gt__(self, other):
        if not self.is_polynomial():
            return True
        elif not other.is_polynomial():
            return False
        else:
            return self.degree > other.degree


class PolynomialDegreeVisitor(ForwardVisitor):
    def register_handlers(self):
        return {
            dex.Variable: self.visit_variable,
            dex.Constant: self.visit_constant,
            dex.Constraint: self.visit_constraint,
            dex.Objective: self.visit_objective,
            dex.ProductExpression: self.visit_product,
            dex.DivisionExpression: self.visit_division,
            dex.LinearExpression: self.visit_linear,
            dex.PowExpression: self.visit_pow,
            dex.SumExpression: self.visit_sum,
            dex.NegationExpression: self.visit_negation,
            dex.UnaryFunctionExpression: self.visit_unary_function,
        }

    def handle_result(self, expr, result, ctx):
        ctx.polynomial[expr] = result
        return True

    def visit_variable(self, _variable, _ctx):
        return PolynomialDegree(1)

    def visit_constant(self, _constant, _ctx):
        return PolynomialDegree(0)

    def visit_constraint(self, expr, ctx):
        return ctx.polynomial[expr.children[0]]

    def visit_objective(self, expr, ctx):
        return ctx.polynomial[expr.children[0]]

    def visit_product(self, expr, ctx):
        return sum([ctx.polynomial[a] for a in expr.children], PolynomialDegree(0))

    def visit_division(self, expr, ctx):
        assert len(expr.children) == 2
        den_degree = ctx.polynomial[expr.children[1]]
        if den_degree.is_polynomial() and den_degree.degree == 0:
            return ctx.polynomial[expr.children[0]]
        return PolynomialDegree.not_polynomial()

    def visit_linear(self, expr, _ctx):
        if len(expr.children) == 0:
            return PolynomialDegree(0)
        return PolynomialDegree(1)

    def visit_pow(self, expr, ctx):
        assert len(expr.children) == 2
        base, expo = expr.children
        base_degree = ctx.polynomial[base]
        expo_degree = ctx.polynomial[expo]

        if not (base_degree.is_polynomial() and expo_degree.is_polynomial()):
            return PolynomialDegree.not_polynomial()

        if expo_degree.degree == 0:
            if base_degree.degree == 0:
                # const ** const
                return PolynomialDegree(0)
            if not isinstance(expo, dex.Constant):
                return PolynomialDegree.not_polynomial()
            expo = expo.value
            if expo == int(expo):
                return base_degree ** expo
        return PolynomialDegree.not_polynomial()

    def visit_sum(self, expr, ctx):
        return max(ctx.polynomial[a] for a in expr.children)

    def visit_negation(self, expr, ctx):
        child = expr.children[0]
        return ctx.polynomial[child]

    def visit_unary_function(self, _expr, _ctx):
        return PolynomialDegree.not_polynomial()

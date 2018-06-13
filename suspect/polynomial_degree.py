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

from typing import Dict
import suspect.dag.expressions as dex
from suspect.dag.iterator import DagForwardIterator
from suspect.dag.visitor import ForwardVisitor


def polynomial_degree(dag, ctx):
    """Compute polynomial degree of expressions in dag."""
    iterator = DagForwardIterator()
    iterator.iterate(dag, PolynomialDegreeVisitor(), ctx)
    return ctx


class PolynomialDegree(object):
    """Represent a polynomial degree."""
    def __init__(self, degree):
        self.degree = degree

    @classmethod
    def not_polynomial(cls):
        """Constructor for non polynomial polynomial."""
        return PolynomialDegree(None)

    def is_polynomial(self):
        """Predicate to check if it's polynomial."""
        return self.degree is not None

    def is_linear(self):
        """Predicate to check if it's linear."""
        return self.is_polynomial() and self.degree == 1

    def is_quadratic(self):
        """Predicate to check if it's quadratic."""
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
        if not other.is_polynomial():
            return False
        return self.degree > other.degree

    def __str__(self):
        return 'PolynomialDegree(degree={})'.format(self.degree)

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))


class PolynomialDegreeVisitor(ForwardVisitor[PolynomialDegree,
                                             Dict[dex.Expression, PolynomialDegree]]):
    """Visitor to compute polynomial degree of expressions."""
    def register_callbacks(self):
        return {
            dex.Variable: self._visit_variable,
            dex.Constant: self._visit_constant,
            dex.Constraint: self._visit_constraint,
            dex.Objective: self._visit_objective,
            dex.ProductExpression: self._visit_product,
            dex.DivisionExpression: self._visit_division,
            dex.LinearExpression: self._visit_linear,
            dex.PowExpression: self._visit_pow,
            dex.SumExpression: self._visit_sum,
            dex.NegationExpression: self._visit_negation,
            dex.UnaryFunctionExpression: self._visit_unary_function,
        }

    def handle_result(self, expr, result, ctx):
        ctx[expr] = result
        return True

    def _visit_variable(self, _variable, _ctx):
        return PolynomialDegree(1)

    def _visit_constant(self, _constant, _ctx):
        return PolynomialDegree(0)

    def _visit_constraint(self, expr, ctx):
        return ctx[expr.children[0]]

    def _visit_objective(self, expr, ctx):
        return ctx[expr.children[0]]

    def _visit_product(self, expr, ctx):
        return sum([ctx[a] for a in expr.children], PolynomialDegree(0))

    def _visit_division(self, expr, ctx):
        assert len(expr.children) == 2
        den_degree = ctx[expr.children[1]]
        if den_degree.is_polynomial() and den_degree.degree == 0:
            # constant denominator
            return ctx[expr.children[0]]
        return PolynomialDegree.not_polynomial()

    def _visit_linear(self, expr, _ctx):
        if not expr.children:
            return PolynomialDegree(0)
        return PolynomialDegree(1)

    def _visit_pow(self, expr, ctx):
        assert len(expr.children) == 2
        base, expo = expr.children
        base_degree = ctx[base]
        expo_degree = ctx[expo]

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

    def _visit_sum(self, expr, ctx):
        return max(ctx[a] for a in expr.children)

    def _visit_negation(self, expr, ctx):
        child = expr.children[0]
        return ctx[child]

    def _visit_unary_function(self, _expr, _ctx):
        return PolynomialDegree.not_polynomial()

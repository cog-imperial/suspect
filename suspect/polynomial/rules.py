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

"""Polynomiality rules."""
from suspect.polynomial.degree import PolynomialDegree
from suspect.expression import ExpressionType
from suspect.interfaces import Rule

__all__ = [
    'VariableRule', 'ConstantRule', 'ConstraintRule', 'ObjectiveRule', 'DivisionRule',
    'ProductRule', 'LinearRule', 'SumRule', 'NegationRule', 'PowerRule', 'UnaryFunctionRule',
]


class VariableRule(Rule):
    """Return polynomial degree of variable."""
    root_expr = ExpressionType.Variable

    def apply(self, _expr, _ctx):
        return PolynomialDegree(1)


class ConstantRule(Rule):
    """Return polynomial degree of constant."""
    root_expr = ExpressionType.Constant

    def apply(self, _expr, _ctx):
        return PolynomialDegree(0)


class ConstraintRule(Rule):
    """Return polynomial degree of constraint."""
    root_expr = ExpressionType.Constraint

    def apply(self, expr, ctx):
        return ctx[expr.children[0]]


class ObjectiveRule(Rule):
    """Return polynomial degree of objective."""
    root_expr = ExpressionType.Objective

    def apply(self, expr, ctx):
        return ctx[expr.children[0]]


class DivisionRule(Rule):
    """Return polynomial degree of division.

    If the denominator is constant, it returns the polynomial degree of the numerator,
    otherwise return no polynomial degree.
    """
    root_expr = ExpressionType.Division

    def apply(self, expr, ctx):
        den_degree = ctx[expr.children[1]]
        if den_degree.is_polynomial() and den_degree.degree == 0:
            # constant denominator
            return ctx[expr.children[0]]
        return PolynomialDegree.not_polynomial()


class ProductRule(Rule):
    """Return polynomial degree of product.

    The polynomial degree of a product is equal to the sum of the polynomial degre of its children.
    """
    root_expr = ExpressionType.Product

    def apply(self, expr, ctx):
        return sum([ctx[a] for a in expr.children], PolynomialDegree(0))


class LinearRule(Rule):
    """Return polynomial degree of a linear expression."""
    root_expr = ExpressionType.Linear

    def apply(self, expr, ctx):
        if not expr.children:
            return PolynomialDegree(0)
        return PolynomialDegree(1)


class SumRule(Rule):
    """Return polynomial degree of a summation expression."""
    root_expr = ExpressionType.Sum

    def apply(self, expr, ctx):
        return max(ctx[a] for a in expr.children)


class NegationRule(Rule):
    """Return polynomial degree of a negation expression."""
    root_expr = ExpressionType.Negation

    def apply(self, expr, ctx):
        return ctx[expr.children[0]]


class PowerRule(Rule):
    """Return polynomial degree of a power expression."""
    root_expr = ExpressionType.Power

    def apply(self, expr, ctx):
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
            if not expo.is_constant():
                return PolynomialDegree.not_polynomial()
            expo = expo.value
            if expo == int(expo):
                return base_degree ** expo
        return PolynomialDegree.not_polynomial()


class UnaryFunctionRule(Rule):
    """Return polynomial degree of an unary function."""
    root_expr = ExpressionType.UnaryFunction

    def apply(self, expr, ctx):
        return PolynomialDegree.not_polynomial()

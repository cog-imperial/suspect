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
from suspect.pyomo.expressions import nonpyomo_leaf_types
from suspect.polynomial.degree import PolynomialDegree
from suspect.interfaces import Rule


class PolynomialDegreeRule(Rule):
    """Base class for rules that compute polynomial degree
     properties of an expression."""
    def apply(self, expr, polynomial_degree):
        """Apply rule to `expr`.

        Parameters
        ----------
        expr : Expression
            the expression
        polynomial_degree : dict-like
            contains polynomial degree of children
        """
        raise NotImplementedError('apply')


class VariableRule(PolynomialDegreeRule):
    """Return polynomial degree of variable."""
    def apply(self, _expr, _poly):
        return PolynomialDegree(1)


class ConstantRule(PolynomialDegreeRule):
    """Return polynomial degree of constant."""
    def apply(self, _expr, _poly):
        return PolynomialDegree(0)


class ConstraintRule(PolynomialDegreeRule):
    """Return polynomial degree of constraint."""
    def apply(self, expr, poly):
        return poly[expr.args[0]]


class ObjectiveRule(PolynomialDegreeRule):
    """Return polynomial degree of objective."""
    def apply(self, expr, poly):
        return poly[expr.args[0]]


class DivisionRule(PolynomialDegreeRule):
    """Return polynomial degree of division.

    If the denominator is constant, it returns the polynomial degree of the numerator,
    otherwise return no polynomial degree.
    """
    def apply(self, expr, poly):
        den_degree = poly[expr.args[1]]
        if den_degree.is_polynomial() and den_degree.degree == 0:
            # constant denominator
            return poly[expr.args[0]]
        return PolynomialDegree.not_polynomial()


class ReciprocalRule(PolynomialDegreeRule):
    """Return polynomial degree of reciprocal."""
    def apply(self, expr, poly):
        den_degree = poly[expr.args[0]]
        if den_degree.is_polynomial() and den_degree.degree == 0:
            return PolynomialDegree(0)
        return PolynomialDegree.not_polynomial()


class ProductRule(PolynomialDegreeRule):
    """Return polynomial degree of product.

    The polynomial degree of a product is equal to the sum of the polynomial degre of its children.
    """
    def apply(self, expr, poly):
        return sum([poly[a] for a in expr.args], PolynomialDegree(0))


class LinearRule(PolynomialDegreeRule):
    """Return polynomial degree of a linear expression."""
    def apply(self, expr, poly):
        if not expr.args:
            return PolynomialDegree(0)
        return PolynomialDegree(1)


class QuadraticRule(PolynomialDegreeRule):
    """Return polynomial degree of a quadratic expression."""
    def apply(self, expr, _poly):
        if not expr.args:
            return PolynomialDegree(0)
        return PolynomialDegree(2)


class SumRule(PolynomialDegreeRule):
    """Return polynomial degree of a summation expression."""
    def apply(self, expr, poly):
        return max(poly[a] for a in expr.args)


class NegationRule(PolynomialDegreeRule):
    """Return polynomial degree of a negation expression."""
    def apply(self, expr, poly):
        return poly[expr.args[0]]


class AbsRule(PolynomialDegreeRule):
    """Return polynomial degree of a abs expression."""
    def apply(self, expr, poly):
        return poly[expr.args[0]]


class PowerRule(PolynomialDegreeRule):
    """Return polynomial degree of a power expression."""
    def apply(self, expr, poly):
        assert len(expr.args) == 2
        base, expo = expr.args
        base_degree = poly[base]
        expo_degree = poly[expo]

        if not (base_degree.is_polynomial() and expo_degree.is_polynomial()):
            return PolynomialDegree.not_polynomial()

        if expo_degree.degree == 0:
            if base_degree.degree == 0:
                # const ** const
                return PolynomialDegree(0)
            if type(expo) not in nonpyomo_leaf_types:
                if not expo.is_constant():
                    return PolynomialDegree.not_polynomial()
                expo = expo.value
            if expo == int(expo):
                return base_degree ** expo
        return PolynomialDegree.not_polynomial()


class UnaryFunctionRule(PolynomialDegreeRule):
    """Return polynomial degree of an unary function."""
    def apply(self, expr, poly):
        return PolynomialDegree.not_polynomial()

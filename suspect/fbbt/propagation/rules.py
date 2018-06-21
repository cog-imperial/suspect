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

"""FBBT bounds propagation rules."""
import operator
from functools import reduce
from suspect.interval import Interval
from suspect.interfaces import Rule
from suspect.expression import ExpressionType, UnaryFunctionType
from suspect.math import almosteq # pylint: disable=no-name-in-module


class VariableRule(Rule):
    """Bound propagation rule for variables."""
    root_expr = ExpressionType.Variable

    def apply(self, expr, _ctx):
        return Interval(expr.lower_bound, expr.upper_bound)


class ConstantRule(Rule):
    """Bound propagation rule for constants."""
    root_expr = ExpressionType.Constant

    def apply(self, expr, _ctx):
        return Interval(expr.value, expr.value)


class ConstraintRule(Rule):
    """Bound propagation rule for constraints."""
    root_expr = ExpressionType.Constraint

    def apply(self, expr, ctx):
        child = expr.children[0]
        child_bounds = ctx.bounds(child)
        constraint_bounds = Interval(expr.lower_bound, expr.upper_bound)
        return constraint_bounds.intersect(child_bounds)


class ObjectiveRule(Rule):
    """Bound propagation rule for objectives."""
    root_expr = ExpressionType.Objective

    def apply(self, expr, ctx):
        child = expr.children[0]
        return ctx.bounds(child)


class ProductRule(Rule):
    """Bound propagation rule for products."""
    root_expr = ExpressionType.Product

    def apply(self, expr, ctx):
        children_bounds = [ctx.bounds(child) for child in expr.children]
        return reduce(operator.mul, children_bounds, 1.0)


class DivisionRule(Rule):
    """Bound propagation rule for divisions."""
    root_expr = ExpressionType.Division

    def apply(self, expr, ctx):
        num, den = expr.children
        return ctx.bounds(num) / ctx.bounds(den)


class LinearRule(Rule):
    """Bound propagation rule for linear expressions."""
    root_expr = ExpressionType.Linear

    def apply(self, expr, ctx):
        children_contribution = sum(
            coef * ctx.bounds(child)
            for coef, child in zip(expr.coefficients, expr.children)
        )
        constant_contribution = Interval(expr.constant_term, expr.constant_term)
        return children_contribution + constant_contribution


class SumRule(Rule):
    """Bound propagation rule for sum."""
    root_expr = ExpressionType.Sum

    def apply(self, expr, ctx):
        return sum(ctx.bounds(child) for child in expr.children)


class PowerRule(Rule):
    """Bound propagation rule for power."""
    root_expr = ExpressionType.Power

    def apply(self, expr, ctx):
        _, expo = expr.children
        if not expo.is_constant():
            return Interval(None, None)

        is_even = almosteq(expo.value % 2, 0)
        is_positive = expo.value > 0
        if is_even and is_positive:
            return Interval(0, None)
        return Interval(None, None)


class NegationRule(Rule):
    """Bound propagation rule for negation."""
    root_expr = ExpressionType.Negation

    def apply(self, expr, ctx):
        return -ctx.bounds(expr.children[0])


class UnaryFunctionRule(Rule):
    """Bound propagation rule for unary functions."""
    root_expr = ExpressionType.UnaryFunction

    _FUNC_TYPE_TO_NAME = {
        UnaryFunctionType.Abs: 'abs',
        UnaryFunctionType.Sqrt: 'sqrt',
        UnaryFunctionType.Exp: 'exp',
        UnaryFunctionType.Log: 'log',
        UnaryFunctionType.Sin: 'sin',
        UnaryFunctionType.Cos: 'cos',
        UnaryFunctionType.Tan: 'tan',
        UnaryFunctionType.Asin: 'asin',
        UnaryFunctionType.Acos: 'acos',
        UnaryFunctionType.Atan: 'atan',
    }

    def apply(self, expr, ctx):
        child = expr.children[0]
        child_bounds = ctx.bounds(child)
        func_name = self._func_name(expr.func_type)
        func = getattr(child_bounds, func_name)
        return func()

    def _func_name(self, func_type):
        func_name = self._FUNC_TYPE_TO_NAME.get(func_type)
        if func_name is None:
            raise RuntimeError('Invalid function type {}'.format(func_type))
        return func_name

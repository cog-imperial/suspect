# Copyright 2017 Francesco Ceccon
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

"""FBBT bounds tightening rules."""
from suspect.interfaces import Rule, UnaryFunctionRule
from suspect.expression import ExpressionType, UnaryFunctionType
from suspect.interval import Interval
from suspect.math import almosteq, inf # pylint: disable=no-name-in-module


MAX_EXPR_CHILDREN = 1000


class ConstraintRule(Rule):
    """Return new bounds for constraint."""
    root_expr = ExpressionType.Constraint

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = Interval(expr.lower_bound, expr.upper_bound)
        return {
            child: bounds
        }


class SumRule(Rule):
    """Return new bounds for sum."""
    root_expr = ExpressionType.Sum

    def apply(self, expr, ctx):
        expr_bound = ctx.bounds(expr)
        if expr_bound.size() == inf:
            return None
        if len(expr.children) > MAX_EXPR_CHILDREN: # pragma: no cover
            return None
        child_bounds = {}
        for child, siblings in _sum_child_and_siblings(expr.children):
            siblings_bound = sum(ctx.bounds(s) for s in siblings)
            child_bounds[child] = expr_bound - siblings_bound
        return child_bounds


class LinearRule(Rule):
    """Return new bounds for linear expressions."""
    root_expr = ExpressionType.Linear

    def apply(self, expr, ctx):
        expr_bound = ctx.bounds(expr)
        if expr_bound.size() == inf:
            return None
        if len(expr.children) > MAX_EXPR_CHILDREN: # pragma: no cover
            return None
        child_bounds = {}
        const = expr.constant_term
        for (child_c, child), siblings in _linear_child_and_siblings(expr.coefficients, expr.children):
            siblings_bound = sum(ctx.bounds(s) * c for c, s in siblings) + const
            child_bounds[child] = (expr_bound - siblings_bound) / child_c
        return child_bounds


class PowerRule(Rule):
    """Return new bounds for power expressions."""
    root_expr = ExpressionType.Power

    def apply(self, expr, ctx):
        base, expo = expr.children
        if not expo.is_constant():
            return None
        if not almosteq(expo.value, 2):
            return None

        bounds = ctx.bounds(expr)
        # the bound of a square number is never negative, but check anyway to
        # avoid unexpected crashes.
        if not bounds.is_nonnegative():
            return None

        sqrt_bound = bounds.sqrt()
        return {
            base: Interval(-sqrt_bound.upper_bound, sqrt_bound.upper_bound)
        }


class _UnaryFunctionRule(UnaryFunctionRule):
    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(expr)
        return {
            child: self._child_bounds(bounds)
        }

    def _child_bounds(self, bounds):
        pass


class _BoundedFunctionRule(_UnaryFunctionRule):
    func_name = None

    def _child_bounds(self, bounds):
        func = getattr(bounds, self.func_name)
        return func.inverse()


class AbsRule(_BoundedFunctionRule):
    """Return new bounds for abs."""
    func_name = 'abs'


class SqrtRule(_BoundedFunctionRule):
    """Return new bounds for sqrt."""
    func_name = 'sqrt'


class ExpRule(_BoundedFunctionRule):
    """Return new bounds for exp."""
    func_name = 'exp'


class LogRule(_BoundedFunctionRule):
    """Return new bounds for log."""
    func_name = 'log'


class _UnboundedFunctionRule(_UnaryFunctionRule):
    def _child_bounds(self, bounds):
        return Interval(None, None)


class SinRule(_UnboundedFunctionRule):
    """Return new bounds for sin."""
    func_type = UnaryFunctionType.Sin


class CosRule(_UnboundedFunctionRule):
    """Return new bounds for cos."""
    func_type = UnaryFunctionType.Cos


class TanRule(_UnboundedFunctionRule):
    """Return new bounds for tan."""
    func_type = UnaryFunctionType.Tan


class AsinRule(_UnboundedFunctionRule):
    """Return new bounds for asin."""
    func_type = UnaryFunctionType.Asin


class AcosRule(_UnboundedFunctionRule):
    """Return new bounds for acos."""
    func_type = UnaryFunctionType.Acos


class AtanRule(_UnboundedFunctionRule):
    """Return new bounds for atan."""
    func_type = UnaryFunctionType.Atan


def _sum_child_and_siblings(children):
    for i, _ in enumerate(children):
        yield children[i], children[:i] + children[i+1:]


def _linear_child_and_siblings(coefficients, children):
    for i, child in enumerate(children):
        child_c = coefficients[i]
        other_children = children[:i] + children[i+1:]
        other_coefficients = coefficients[:i] + coefficients[i+1:]
        yield (child_c, child), zip(other_coefficients, other_children)

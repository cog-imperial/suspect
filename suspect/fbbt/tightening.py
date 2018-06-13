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

"""FBBT bounds tightening."""
from typing import Dict
import suspect.dag.expressions as dex
from suspect.dag.visitor import BackwardVisitor
from suspect.interval import Interval
from suspect.math import inf, almosteq # pylint: disable=no-name-in-module


MAX_EXPR_CHILDREN = 1000


class BoundsTighteningVisitor(BackwardVisitor[Interval, Dict[dex.Expression, Interval]]):
    """Tighten bounds from sinks to sources."""
    def register_callbacks(self):
        return {
            dex.Constraint: self._visit_constraint,
            dex.SumExpression: self._visit_sum,
            dex.LinearExpression: self._visit_linear,
            dex.PowExpression: self._visit_pow,
            dex.UnaryFunctionExpression: self._visit_unary_function,
        }

    def handle_result(self, expr, value, bounds):
        if value is None:
            new_bound = Interval(None, None)
        else:
            new_bound = value

        if expr in bounds:
            old_bound = bounds[expr]
            new_bound = old_bound.intersect(new_bound)
            has_changed = old_bound != new_bound
        else:
            has_changed = True

        bounds[expr] = new_bound
        # return has_changed
        return True

    def _visit_constraint(self, expr, _bounds):
        child = expr.children[0]
        bound = Interval(expr.lower_bound, expr.upper_bound)
        return {
            child: bound
        }

    def _visit_sum(self, expr, bounds):
        expr_bound = bounds[expr]
        if expr_bound.size() == inf:
            return None
        if len(expr.children) > MAX_EXPR_CHILDREN:
            return None
        child_bounds = {}
        for child, siblings in _sum_child_and_siblings(expr.children):
            siblings_bound = sum(bounds[s] for s in siblings)
            child_bounds[child] = expr_bound - siblings_bound
        return child_bounds

    def _visit_linear(self, expr, bounds):
        expr_bound = bounds[expr]
        if expr_bound.size() == inf:
            return None
        if len(expr.children) > MAX_EXPR_CHILDREN:
            return None
        child_bounds = {}
        const = expr.constant_term
        for (child_c, child), siblings in _linear_child_and_siblings(expr.coefficients, expr.children):
            siblings_bound = sum(bounds[s] * c for c, s in siblings) + const
            child_bounds[child] = (expr_bound - siblings_bound) / child_c
        return child_bounds

    def _visit_pow(self, expr, bounds):
        base, expo = expr.children
        if not isinstance(expo, dex.Constant):
            return None
        if not almosteq(expo.value, 2):
            return None

        bound = bounds[expr]
        # the bound of a square number is never negative, but check anyway to
        # avoid unexpected crashes.
        if not bound.is_nonnegative():
            return None

        sqrt_bound = bound.sqrt()
        return {
            base: Interval(-sqrt_bound.upper_bound, sqrt_bound.upper_bound)
        }

    def _visit_unary_function(self, expr, bounds):
        child = expr.children[0]
        bound = bounds[expr]
        if expr.func_name in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
            return {
                child: Interval(None, None)
            }
        func = getattr(bound, expr.func_name)
        return {
            child: func.inverse()
        }


def _sum_child_and_siblings(children):
    for i, _ in enumerate(children):
        yield children[i], children[:i] + children[i+1:]


def _linear_child_and_siblings(coefficients, children):
    for i, child in enumerate(children):
        child_c = coefficients[i]
        other_children = children[:i] + children[i+1:]
        other_coefficients = coefficients[:i] + coefficients[i+1:]
        yield (child_c, child), zip(other_coefficients, other_children)

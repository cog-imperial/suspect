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

"""FBBT bounds propagation."""
import operator
from functools import reduce
import suspect.dag.expressions as dex
from suspect.dag.visitor import ForwardVisitor
from suspect.interval import Interval
from suspect.math import almosteq # pylint: disable=no-name-in-module


class BoundsPropagationVisitor(ForwardVisitor):
    """Propagate bounds from sources to sinks."""
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
            dex.UnaryFunctionExpression: self._visit_unary_function,
        }

    def handle_result(self, expr, new_bound, bounds):
        if expr in bounds:
            # bounds exists, tighten it
            old_bound = bounds[expr]
            new_bound = old_bound.intersect(new_bound)
            has_changed = new_bound != old_bound
        else:
            has_changed = True
        bounds[expr] = new_bound
        # return has_changed
        return True

    def _visit_variable(self, var, _bounds):
        return Interval(var.lower_bound, var.upper_bound)

    def _visit_constant(self, const, _bounds):
        return Interval(const.value, const.value)

    def _visit_constraint(self, constr, bounds):
        arg = constr.children[0]
        inner = bounds[arg]
        new_lb = max(inner.lower_bound, constr.lower_bound)
        new_ub = min(inner.upper_bound, constr.upper_bound)
        if not new_lb <= new_ub:
            raise RuntimeError('Infeasible bound.')
        return Interval(new_lb, new_ub)

    def _visit_objective(self, obj, bounds):
        arg = obj.children[0]
        return bounds[arg]

    def _visit_product(self, expr, bounds):
        children_bounds = [bounds[c] for c in expr.children]
        return reduce(operator.mul, children_bounds, 1)

    def _visit_division(self, expr, bounds):
        top, bot = expr.children
        return bounds[top] / bounds[bot]

    def _visit_linear(self, expr, bounds):
        children_bounds = [
            coef * bounds[c]
            for coef, c in zip(expr.coefficients, expr.children)
        ]
        children_bounds.append(Interval(expr.constant_term, expr.constant_term))
        return sum(children_bounds)

    def _visit_sum(self, expr, bounds):
        children_bounds = [bounds[c] for c in expr.children]
        return sum(children_bounds)

    def _visit_pow(self, expr, _bounds):
        _, expo = expr.children
        if not isinstance(expo, dex.Constant):
            return Interval(None, None)
        is_even = almosteq(expo.value % 2, 0)
        is_positive = expo.value > 0
        if is_even and is_positive:
            return Interval(0, None)
        return Interval(None, None)

    def _visit_unary_function(self, expr, bounds):
        arg_bound = bounds[expr.children[0]]
        func = getattr(arg_bound, expr.func_name)
        return func()

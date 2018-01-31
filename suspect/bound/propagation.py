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

import operator
from functools import reduce
import suspect.dag.expressions as dex
from suspect.dag.visitor import Dispatcher
from suspect.bound import ArbitraryPrecisionBound as Bound


class BoundsPropagationVisitor(object):
    def __init__(self, ctx):
        self._ctx = ctx
        self._dispatcher = Dispatcher(
            lookup={
                dex.Variable: self.visit_variable,
                dex.Constant: self.visit_constant,
                dex.Constraint: self.visit_constraint,
                dex.Objective: self.visit_objective,
                dex.ProductExpression: self.visit_product,
                dex.DivisionExpression: self.visit_division,
                dex.LinearExpression: self.visit_linear,
                dex.PowExpression: self.visit_pow,
                dex.SumExpression: self.visit_sum,
                dex.UnaryFunctionExpression: self.visit_unary_function,
            },
            allow_missing=False)

    def get(self, expr):
        return self._ctx[id(expr)]

    def set(self, expr, bound):
        self._ctx[id(expr)] = bound

    def visit_variable(self, var):
        return Bound(var.lower_bound, var.upper_bound)

    def visit_constant(self, const):
        return Bound(const.value, const.value)

    def visit_constraint(self, constr):
        arg = constr.children[0]
        inner = self.get(arg)
        new_lb = max(inner.lower_bound, constr.lower_bound)
        new_ub = min(inner.upper_bound, constr.upper_bound)
        if not new_lb <= new_ub:
            raise RuntimeError('Infeasible bound.')
        return Bound(new_lb, new_ub)

    def visit_objective(self, obj):
        arg = obj.children[0]
        return self.get(arg)

    def visit_product(self, expr):
        bounds = [self.get(c) for c in expr.children]
        return reduce(operator.mul, bounds, 1)

    def visit_division(self, expr):
        top, bot = expr.children
        return self.get(top) / self.get(bot)

    def visit_linear(self, expr):
        bounds = [coef * self.get(c) for coef, c in zip(expr.coefficients, expr.children)]
        bounds.append(Bound(expr.constant_term, expr.constant_term))
        return sum(bounds)

    def visit_sum(self, expr):
        bounds = [self.get(c) for c in expr.children]
        return sum(bounds)

    def visit_pow(self, expr):
        return Bound(None, None)

    def visit_unary_function(self, expr):
        arg_bound = self.get(expr.children[0])
        func = getattr(arg_bound, expr.func_name)
        return func()

    def __call__(self, expr):
        new_bound = self._dispatcher.dispatch(expr)
        if new_bound is None:
            raise RuntimeError('new_bound is None.')
        self.set(expr, new_bound)


def propagate_bounds(dag, ctx):
    """Propagate bounds from sources to sinks.

    Arguments
    ---------
    dag: ProblemDag
      the problem
    ctx: dict-like
      the context containing each node bounds
    """
    visitor = BoundsPropagationVisitor(ctx)
    dag.forward_visit(visitor)

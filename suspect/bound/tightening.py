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

import suspect.dag.expressions as dex
from suspect.dag.visitor import BackwardVisitor
from suspect.bound import ArbitraryPrecisionBound as Bound


def tighten_bounds(dag, ctx):
    """Tighten bounds from sinks to sources.

    Parameters
    ----------
    dag: ProblemDag
      the problem
    ctx: dict-like
      the context containing each node bounds
    """
    visitor = BoundsTighteningVisitor()
    dag.backward_visit(visitor, ctx)
    return ctx


class BoundsTighteningVisitor(BackwardVisitor):
    def register_handlers(self):
        return {
            dex.Constraint: self.visit_constraint,
            dex.SumExpression: self.visit_sum,
            dex.LinearExpression: self.visit_linear,
            dex.UnaryFunctionExpression: self.visit_unary_function,
        }

    def handle_result(self, expr, value, ctx):
        if value is None:
            new_bound = Bound(None, None)
        else:
            new_bound = value

        if expr in ctx.bound:
            old_bound = ctx.bound[expr]
            new_bound = old_bound.tighten(new_bound)

        ctx.bound[expr] = new_bound

    def visit_constraint(self, expr, ctx):
        child = expr.children[0]
        bound = Bound(expr.lower_bound, expr.upper_bound)
        return {
            child: bound
        }

    def visit_sum(self, expr, ctx):
        expr_bound = ctx.bound[expr]
        bounds = {}
        for child, siblings in _sum_child_and_siblings(expr.children):
            siblings_bound = sum(ctx.bound[s] for s in siblings)
            bounds[child] = expr_bound - siblings_bound
        return bounds

    def visit_linear(self, expr, ctx):
        expr_bound = ctx.bound[expr]
        bounds = {}
        const = expr.constant_term
        for (child_c, child), siblings in _linear_child_and_siblings(expr.coefficients, expr.children):
            siblings_bound = sum(ctx.bound[s] * c for c, s in siblings) + const
            bounds[child] = (expr_bound - siblings_bound) / child_c
        return bounds

    def visit_unary_function(self, expr, ctx):
        child = expr.children[0]
        bound = ctx.bound[expr]
        func = getattr(bound, expr.func_name)
        return {
            child: func.inv()
        }


def _sum_child_and_siblings(children):
    for i in range(len(children) - 1):
        yield children[i], children[:i] + children[i+1:]


def _linear_child_and_siblings(coefficients, children):
    for i in range(len(children) - 1):
        child = children[i]
        child_c = coefficients[i]
        other_children = children[:i] + children[i+1:]
        other_coefficients = coefficients[:i] + coefficients[i+1:]
        yield (child_c, child), zip(other_coefficients, other_children)

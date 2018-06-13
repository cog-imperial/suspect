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
from suspect.dag.visitor import ForwardVisitor
from suspect.context import SpecialStructurePropagationContext
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.product import product_monotonicity
from suspect.monotonicity.division import division_monotonicity
from suspect.monotonicity.linear import linear_monotonicity
from suspect.monotonicity.pow import pow_monotonicity


class MonotonicityPropagationVisitor(ForwardVisitor):
    def register_callbacks(self):
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

            dex.NegationExpression: self.visit_nonincreasing_function,
            dex.AbsExpression: self.visit_abs,
            dex.SqrtExpression: self.visit_nondecreasing_function,
            dex.ExpExpression: self.visit_nondecreasing_function,
            dex.LogExpression: self.visit_nondecreasing_function,
            dex.SinExpression: self.visit_sin,
            dex.CosExpression: self.visit_cos,
            dex.TanExpression: self.visit_nondecreasing_function,
            dex.AsinExpression: self.visit_nondecreasing_function,
            dex.AcosExpression: self.visit_nonincreasing_function,
            dex.AtanExpression: self.visit_nondecreasing_function,
        }

    def handle_result(self, expr, result, ctx):
        if result is None:
            return False
        ctx.monotonicity[expr] = result
        return not result.is_unknown()

    def visit_variable(self, _expr, _ctx):
        return Monotonicity.Nondecreasing

    def visit_constant(self, _expr, _ctx):
        return Monotonicity.Constant

    def visit_constraint(self, expr, ctx):
        if expr.bounded_below() and expr.bounded_above():
            # l <= g(x) <= u
            mono = ctx.monotonicity[expr.children[0]]
            if mono.is_constant():
                return mono
            else:
                return Monotonicity.Unknown
        elif expr.bounded_below():
            # l <= g(x) => -g(x) <= -l
            mono = ctx.monotonicity[expr.children[0]]
            return mono.negate()
        elif expr.bounded_above():
            # g(x) <= u
            return ctx.monotonicity[expr.children[0]]
        else:
            raise RuntimeError('Constraint with no bounds')

    def visit_objective(self, expr, ctx):
        if expr.sense == dex.Sense.MINIMIZE:
            return ctx.monotonicity[expr.children[0]]
        else:
            # max f(x) == min -f(x)
            mono = ctx.monotonicity[expr.children[0]]
            return mono.negate()

    def visit_product(self, expr, ctx):
        return product_monotonicity(expr, ctx)

    def visit_division(self, expr, ctx):
        return division_monotonicity(expr, ctx)

    def visit_linear(self, expr, ctx):
        return linear_monotonicity(expr, ctx)

    def visit_pow(self, expr, ctx):
        return pow_monotonicity(expr, ctx)

    def visit_sum(self, expr, ctx):
        return linear_monotonicity(expr, ctx)

    def visit_abs(self, expr, ctx):
        arg = expr.children[0]
        mono = ctx.monotonicity[arg]
        bound = ctx.bound[arg]

        # good examples to understand the behaviour of abs are abs(-x) and
        # abs(1/x)
        if bound.is_nonnegative():
            # abs(x), x > 0 is the same as x
            return mono
        elif bound.is_nonpositive():
            # abs(x), x < 0 is the opposite of x
            return mono.negate()
        else:
            return Monotonicity.Unknown

    def visit_nondecreasing_function(self, expr, ctx):
        return ctx.monotonicity[expr.children[0]]

    def visit_nonincreasing_function(self, expr, ctx):
        return ctx.monotonicity[expr.children[0]].negate()

    def visit_sin(self, expr, ctx):
        arg = expr.children[0]
        bound = ctx.bound[arg]
        cos_bound = bound.cos()
        mono = ctx.monotonicity[arg]

        if mono.is_nondecreasing() and cos_bound.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bound.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bound.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nondecreasing() and cos_bound.is_nonpositive():
            return Monotonicity.Nonincreasing
        else:
            return Monotonicity.Unknown

    def visit_cos(self, expr, ctx):
        arg = expr.children[0]
        bound = ctx.bound[arg]
        sin_bound = bound.sin()
        mono = ctx.monotonicity[arg]

        if mono.is_nonincreasing() and sin_bound.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bound.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bound.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nonincreasing() and sin_bound.is_nonpositive():
            return Monotonicity.Nonincreasing
        else:
            return Monotonicity.Unknown

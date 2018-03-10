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
from suspect.convexity.convexity import Convexity
from suspect.convexity.product import product_convexity
from suspect.convexity.division import division_convexity
from suspect.convexity.linear import linear_convexity
from suspect.convexity.pow import pow_convexity
from suspect.convexity.sin import (
    sin_convexity,
    asin_convexity,
)
from suspect.convexity.cos import (
    cos_convexity,
    acos_convexity,
)
from suspect.convexity.tan import (
    tan_convexity,
    atan_convexity,
)


class ConvexityPropagationVisitor(ForwardVisitor):
    def register_handlers(self):
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

            dex.NegationExpression: self.visit_negation,
            dex.AbsExpression: self.visit_abs,
            dex.SqrtExpression: self.visit_sqrt,
            dex.ExpExpression: self.visit_exp,
            dex.LogExpression: self.visit_log,
            dex.SinExpression: self.visit_sin,
            dex.CosExpression: self.visit_cos,
            dex.TanExpression: self.visit_tan,
            dex.AsinExpression: self.visit_asin,
            dex.AcosExpression: self.visit_acos,
            dex.AtanExpression: self.visit_atan,
        }

    def handle_result(self, expr, result, ctx):
        ctx.convexity[expr] = result
        return result.is_unknown()

    def visit_variable(self, _expr, _ctx):
        return Convexity.Linear

    def visit_constant(self, _expr, _ctx):
        return Convexity.Linear

    def visit_constraint(self, expr, ctx):
        cvx = ctx.convexity[expr.children[0]]
        if expr.bounded_below() and expr.bounded_above():
            # l <= g(x) <= u
            if cvx.is_linear():
                return cvx
            else:
                return Convexity.Unknown
        elif expr.bounded_below():
            # l <= g(x) => -g(x) <= -l
            return cvx.negate()
        elif expr.bounded_above():
            # g(x) <= u
            return cvx
        else:
            raise RuntimeError('Constraint with no bounds')

    def visit_objective(self, expr, ctx):
        cvx = ctx.convexity[expr.children[0]]
        if expr.sense == dex.Sense.MINIMIZE:
            return cvx
        else:
            # max f(x) => min -f(x)
            return cvx.negate()

    def visit_product(self, expr, ctx):
        return product_convexity(expr, ctx)

    def visit_division(self, expr, ctx):
        return division_convexity(expr, ctx)

    def visit_sum(self, expr, ctx):
        return linear_convexity(expr, ctx)

    def visit_linear(self, expr, ctx):
        return linear_convexity(expr, ctx)

    def visit_pow(self, expr, ctx):
        return pow_convexity(expr, ctx)

    def visit_negation(self, expr, ctx):
        return ctx.convexity[expr.children[0]].negate()

    def visit_abs(self, expr, ctx):
        arg = expr.children[0]
        cvx = ctx.convexity[arg]
        bound = ctx.bound[arg]
        if cvx.is_linear():
            return Convexity.Convex
        elif cvx.is_convex():
            if bound.is_nonnegative():
                return Convexity.Convex
            elif bound.is_nonpositive():
                return Convexity.Concave
        elif cvx.is_concave():
            if bound.is_nonpositive():
                return Convexity.Convex
            elif bound.is_nonnegative():
                return Convexity.Concave
        return Convexity.Unknown

    def visit_sqrt(self, expr, ctx):
        arg = expr.children[0]
        bound = ctx.bound[arg]
        cvx = ctx.convexity[arg]
        # TODO: handle sqrt(x*x) which is same as x
        if bound.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave
        return Convexity.Unknown

    def visit_exp(self, expr, ctx):
        arg = expr.children[0]
        cvx = ctx.convexity[arg]
        if cvx.is_convex():
            return Convexity.Convex
        return Convexity.Unknown

    def visit_log(self, expr, ctx):
        arg = expr.children[0]
        bound = ctx.bound[arg]
        cvx = ctx.convexity[arg]
        if bound.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave
        return Convexity.Unknown

    def visit_sin(self, expr, ctx):
        return sin_convexity(expr, ctx)

    def visit_cos(self, expr, ctx):
        return cos_convexity(expr, ctx)

    def visit_tan(self, expr, ctx):
        return tan_convexity(expr, ctx)

    def visit_asin(self, expr, ctx):
        return asin_convexity(expr, ctx)

    def visit_acos(self, expr, ctx):
        return acos_convexity(expr, ctx)

    def visit_atan(self, expr, ctx):
        return atan_convexity(expr, ctx)

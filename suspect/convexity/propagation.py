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
from suspect.dag.visitor import Dispatcher
from suspect.convexity.convexity import Convexity
from suspect.convexity.product import product_convexity
from suspect.convexity.division import division_convexity
from suspect.convexity.linear import linear_convexity
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


class ConvexityPropagationVisitor(object):
    def __init__(self, bounds, mono):
        self._bounds = bounds
        self._mono = mono
        self._cvx = {}
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
            },
            allow_missing=False)

    def bound(self, expr):
        return self._bounds[id(expr)]

    def is_nonnegative(self, expr):
        return self.bound(expr).is_nonnegative()

    def is_nonpositive(self, expr):
        return self.bound(expr).is_nonpositive()

    def monotonicity(self, expr):
        return self._mono[id(expr)]

    def convexity(self, expr):
        return self.get(expr)

    def get(self, expr):
        return self._cvx[id(expr)]

    def set(self, expr, cvx):
        assert isinstance(cvx, Convexity)
        self._cvx[id(expr)] = cvx

    def result(self):
        return self._cvx

    def __call__(self, expr):
        new_cvx = self._dispatcher.dispatch(expr)
        if new_cvx is None:
            raise RuntimeError('new_cvx is None')
        self.set(expr, new_cvx)

    def visit_variable(self, expr):
        return Convexity.Linear

    def visit_constant(self, _expr):
        return Convexity.Linear

    def visit_constraint(self, expr):
        cvx = self.get(expr.children[0])
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

    def visit_objective(self, expr):
        cvx = self.get(expr.children[0])
        if expr.sense == dex.Sense.MINIMIZE:
            return cvx
        else:
            # max f(x) => min -f(x)
            return cvx.negate()

    def visit_product(self, expr):
        return product_convexity(self, expr)

    def visit_division(self, expr):
        return division_convexity(self, expr)

    def visit_sum(self, expr):
        return linear_convexity(self, expr)

    def visit_linear(self, expr):
        return linear_convexity(self, expr)

    def visit_pow(self, expr):
        return pow_convexity(self, expr)

    def visit_negation(self, expr):
        return self.get(expr.children[0]).negate()

    def visit_abs(self, expr):
        arg = expr.children[0]
        cvx = self.get(arg)
        bound = self.bound(arg)
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

    def visit_sqrt(self, expr):
        arg = expr.children[0]
        bound = self.bound(arg)
        cvx = self.get(arg)
        # TODO: handle sqrt(x*x) which is same as x
        if bound.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave
        return Convexity.Unknown

    def visit_exp(self, expr):
        arg = expr.children[0]
        cvx = self.get(arg)
        if cvx.is_convex():
            return Convexity.Convex
        return Convexity.Unknown

    def visit_log(self, expr):
        arg = expr.children[0]
        bound = self.bound(arg)
        cvx = self.get(arg)
        if bound.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave
        return Convexity.Unknown

    def visit_sin(self, expr):
        return sin_convexity(self, expr)

    def visit_cos(self, expr):
        return cos_convexity(self, expr)

    def visit_tan(self, expr):
        return tan_convexity(self, expr)

    def visit_asin(self, expr):
        return asin_convexity(self, expr)

    def visit_acos(self, expr):
        return acos_convexity(self, expr)

    def visit_atan(self, expr):
        return atan_convexity(self, expr)

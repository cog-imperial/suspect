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

from convexity_detection.monotonicity import MonotonicityHandler
from convexity_detection.expr_dict import ExpressionDict
from convexity_detection.expr_visitor import (
    ExpressionHandler,
    bottom_up_visit as visit_expression,
    visit_after,
    accumulated,
)
from convexity_detection.convexity.convexity import Convexity
from convexity_detection.convexity.product import product_convexity
from convexity_detection.convexity.division import division_convexity
from convexity_detection.convexity.pow import pow_convexity
from convexity_detection.convexity.unary_function import (
    unary_function_convexity
)
from convexity_detection.convexity.linear import linear_convexity
from convexity_detection.util import numeric_types


class ConvexityHandler(ExpressionHandler):
    def __init__(self, bounds_memo=None):
        self.mono_handler = MonotonicityHandler(bounds_memo=bounds_memo)
        self.memo = ExpressionDict()

    def accumulate(self, expr, cvx):
        if cvx is not None:
            assert(isinstance(cvx, Convexity))
            self.memo[expr] = cvx

    def convexity(self, expr):
        if isinstance(expr, numeric_types):
            return Convexity.Linear
        else:
            return self.memo[expr]

    def monotonicity(self, expr):
        return self.mono_handler.monotonicity(expr)

    def bound(self, expr):
        return self.mono_handler.bound(expr)

    def is_positive(self, expr):
        return self.bound(expr).is_positive()

    def is_nonnegative(self, expr):
        return self.bound(expr).is_nonnegative()

    def is_negative(self, expr):
        return self.bound(expr).is_negative()

    def is_nonpositive(self, expr):
        return self.bound(expr).is_nonpositive()

    def is_zero(self, expr):
        return self.bound(expr).is_zero()

    def visit_number(self, n):
        pass

    def visit_numeric_constant(self, expr):
        pass

    @accumulated
    @visit_after('mono_handler')
    def visit_variable(self, expr):
        return Convexity.Linear

    @accumulated
    @visit_after('mono_handler')
    def visit_equality(self, expr):
        assert(len(expr._args) == 2)
        body, _ = expr._args
        return self.convexity(body)

    @accumulated
    @visit_after('mono_handler')
    def visit_inequality(self, expr):
        raise ValueError('visit_inequality not implemented')

    @accumulated
    @visit_after('mono_handler')
    def visit_product(self, expr):
        return product_convexity(self, expr)

    @accumulated
    @visit_after('mono_handler')
    def visit_division(self, expr):
        return division_convexity(self, expr)

    @accumulated
    @visit_after('mono_handler')
    def visit_sum(self, expr):
        return linear_convexity(self, expr)

    @accumulated
    @visit_after('mono_handler')
    def visit_linear(self, expr):
        return linear_convexity(self, expr)

    @accumulated
    @visit_after('mono_handler')
    def visit_negation(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        cvx = self.convexity(arg)

        if cvx.is_linear():
            return Convexity.Linear
        elif cvx.is_convex():
            return Convexity.Concave
        elif cvx.is_concave():
            return Convexity.Convex

        return Convexity.Unknown

    @accumulated
    @visit_after('mono_handler')
    def visit_unary_function(self, expr):
        return unary_function_convexity(self, expr)

    @accumulated
    @visit_after('mono_handler')
    def visit_abs(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        cvx = self.convexity(arg)

        if cvx.is_linear():
            return Convexity.Convex
        elif cvx.is_convex():
            if self.is_zero(arg):
                return Convexity.Linear
            elif self.is_nonnegative(arg):
                return Convexity.Convex
            elif self.is_nonpositive(arg):
                return Convexity.Concave

        elif cvx.is_concave():
            if self.is_zero(arg):
                return Convexity.Linear
            elif self.is_nonnegative(arg):
                return Convexity.Concave
            elif self.is_nonpositive(arg):
                return Convexity.Convex

        return Convexity.Unknown

    @accumulated
    @visit_after('mono_handler')
    def visit_pow(self, expr):
        return pow_convexity(self, expr)


def expression_convexity(expr, bounds=None):
    handler = ConvexityHandler(bounds_memo=bounds)
    visit_expression(handler, expr)
    return handler.convexity(expr)

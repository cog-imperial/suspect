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

from suspect.bounds import BoundsHandler, expression_bounds
from suspect.expr_dict import ExpressionDict
from suspect.expr_visitor import (
    ExpressionHandler,
    bottom_up_visit as visit_expression,
    visit_after,
    accumulated,
)
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.product import product_monotonicity
from suspect.monotonicity.division import division_monotonicity
from suspect.monotonicity.pow import pow_monotonicity
from suspect.monotonicity.linear import linear_monotonicity
from suspect.util import numeric_types

import pyomo.environ as aml


class MonotonicityHandler(ExpressionHandler):
    def __init__(self, bounds_memo=None):
        self.bounds_handler = BoundsHandler(memo=bounds_memo)
        self.memo = ExpressionDict()

    def accumulate(self, expr, mono):
        if mono is not None:
            self.memo[expr] = mono

    def monotonicity(self, expr):
        if isinstance(expr, numeric_types):
            return Monotonicity.Constant
        else:
            return self.memo[expr]

    def bound(self, expr):
        return self.bounds_handler.bound(expr)

    def is_nonnegative(self, expr):
        return self.bound(expr).is_nonnegative()

    def is_nonpositive(self, expr):
        return self.bound(expr).is_nonpositive()

    def visit_number(self, n):
        pass

    def visit_numeric_constant(self, expr):
        pass

    @accumulated
    @visit_after('bounds_handler')
    def visit_variable(self, expr):
        return Monotonicity.Nondecreasing

    @accumulated
    @visit_after('bounds_handler')
    def visit_equality(self, expr):
        raise RuntimeError('suspect expects problems in standard form')

    @visit_after('bounds_handler')
    def visit_inequality(self, expr):
        assert(len(expr._args) == 2)
        body, bound = expr._args
        assert(isinstance(bound, numeric_types))
        return self.monotonicity(body)

    @accumulated
    @visit_after('bounds_handler')
    def visit_product(self, expr):
        return product_monotonicity(self, expr)

    @accumulated
    @visit_after('bounds_handler')
    def visit_division(self, expr):
        return division_monotonicity(self, expr)

    @accumulated
    @visit_after('bounds_handler')
    def visit_sum(self, expr):
        return linear_monotonicity(self, expr)

    @accumulated
    @visit_after('bounds_handler')
    def visit_linear(self, expr):
        return linear_monotonicity(self, expr)

    @accumulated
    @visit_after('bounds_handler')
    def visit_negation(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        mono = self.monotonicity(arg)

        if mono.is_nondecreasing():
            return Monotonicity.Nonincreasing
        elif mono.is_nonincreasing():
            return Monotonicity.Nondecreasing
        else:
            return Monotonicity.Unknown

    @accumulated
    @visit_after('bounds_handler')
    def visit_unary_function(self, expr):
        assert len(expr._args) == 1

        name = expr._name
        arg = expr._args[0]
        mono = self.monotonicity(arg)

        if name in ['sqrt', 'log', 'asin', 'atan', 'tan', 'exp']:
            return mono

        elif name in ['acos']:
            if mono.is_constant():
                return Monotonicity.Constant
            elif mono.is_nondecreasing():
                return Monotonicity.Nonincreasing
            elif mono.is_nonincreasing():
                return Monotonicity.Nondecreasing
            else:
                return Monotonicity.Unknown

        elif name == 'sin':
            cos_arg = aml.cos(arg)
            cos_bound = expression_bounds(cos_arg)
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

        elif name == 'cos':
            sin_arg = aml.sin(arg)
            sin_bound = expression_bounds(sin_arg)
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

        else:
            raise RuntimeError('unknown unary function {}'.format(name))

    @accumulated
    @visit_after('bounds_handler')
    def visit_abs(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        mono = self.monotonicity(arg)

        # good examples to understand the behaviour of abs are abs(-x) and
        # abs(1/x)
        if self.is_nonnegative(arg):
            # abs(x), x > 0 is the same as x
            return mono
        elif self.is_nonpositive(arg):
            # abs(x), x < 0 is the opposite of x
            if mono.is_nondecreasing():
                return Monotonicity.Nonincreasing
            else:
                return Monotonicity.Nondecreasing
        else:
            return Monotonicity.Unknown


    @accumulated
    @visit_after('bounds_handler')
    def visit_pow(self, expr):
        return pow_monotonicity(self, expr)


def expression_monotonicity(expr):
    handler = MonotonicityHandler()
    visit_expression(handler, expr)
    return handler.monotonicity(expr)


def is_nondecreasing(expr):
    return expression_monotonicity(expr).is_nondecreasing()


def is_nonincreasing(expr):
    return expression_monotonicity(expr).is_nonincreasing()


def is_unknown(expr):
    return expression_monotonicity(expr).is_unknown()

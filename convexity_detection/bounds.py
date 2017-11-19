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

from numbers import Number
import operator
from functools import reduce
import warnings
from convexity_detection.expr_visitor import (
    bottom_up_visit as visit_expression,
    ExpressionHandler,
    accumulated,
)
from convexity_detection.util import (
    numeric_types,
    numeric_value,
)
from convexity_detection.expr_dict import ExpressionDict
from convexity_detection.math import *
from convexity_detection.error import DomainError


class Bound(object):
    def __init__(self, l, u):
        if l is None:
            l = -inf
        if u is None:
            u = inf

        if not isinstance(l, mpf):
            l = mpf(l)
        if not isinstance(u, mpf):
            u = mpf(u)

        if l > u:
            raise ValueError('l must be >= u')

        self.l = l
        self.u = u

    def is_zero(self):
        return almosteq(self.l, 0) and almosteq(self.u, 0)

    def is_positive(self):
        return self.l > 0

    def is_negative(self):
        return self.u < 0

    def is_nonnegative(self):
        return almostgte(self.l, 0)

    def is_nonpositive(self):
        return almostlte(self.u, 0)

    def __add__(self, other):
        l = self.l
        u = self.u
        if isinstance(other, Bound):
            return Bound(l + other.l, u + other.u)
        elif isinstance(other, Number):
            return Bound(l + other, u + other)
        else:
            raise TypeError('adding Bound to incompatbile type')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        l = self.l
        u = self.u
        if isinstance(other, Bound):
            return Bound(l - other.u, u - other.l)
        elif isinstance(other, Number):
            return Bound(l - other, u - other)
        else:
            raise TypeError('subtracting Bound to incompatbile type')

    def __mul__(self, other):
        l = self.l
        u = self.u
        if isinstance(other, Bound):
            ol = other.l
            ou = other.u
            new_l = min(l * ol, l * ou, u * ol, u * ou)
            new_u = max(l * ol, l * ou, u * ol, u * ou)
            return Bound(new_l, new_u)
        elif isinstance(other, Number):
            return self.__mul__(Bound(other, other))
        else:
            raise TypeError('multiplying Bound to incompatible type')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Bound):
            ol = other.l
            ou = other.u
            if ol <= 0 and ou >= 0:
                return Bound(-inf, inf)
            else:
                return self.__mul__(Bound(1/ou, 1/ol))
        elif isinstance(other, Number):
            return self.__truediv__(Bound(other, other))
        else:
            raise TypeError('dividing Bound by incompatible type')

    def __eq__(self, other):
        if not isinstance(other, Bound):
            return False
        return almosteq(self.l, other.l) and almosteq(self.u, other.u)

    def __contains__(self, other):
        return other.l >= self.l and other.u <= self.u

    def __repr__(self):
        return '<{} at {}>'.format(str(self), id(self))

    def __str__(self):
        return '[{}, {}]'.format(self.l, self.u)


def _sin_bound(lower, upper):
    if upper - lower >= 2 * pi:
        return Bound(-1, 1)
    else:
        l = lower % (2 * pi)
        u = l + (upper - lower)
        new_u = max(sin(l), sin(u))
        new_l = min(sin(l), sin(u))
        if l <= 0.5 * pi <= u:
            new_u = 1
        if l <= 1.5 * pi <= u:
            new_l = -1
        return Bound(new_l, new_u)


class BoundsHandler(ExpressionHandler):
    def __init__(self):
        self.memo = ExpressionDict()

    def accumulate(self, expr, bound):
        if bound is not None:
            self.memo[expr] = bound

    def bound(self, expr):
        if isinstance(expr, numeric_types):
            value = numeric_value(expr)
            return Bound(value, value)
        else:
            return self.memo[expr]

    @accumulated
    def visit_variable(self, v):
        return Bound(v.bounds[0], v.bounds[1])

    @accumulated
    def visit_number(self, n):
        pass  # do nothing

    @accumulated
    def visit_numeric_constant(self, n):
        pass

    @accumulated
    def visit_equality(self, expr):
        assert(len(expr._args) == 2)
        _, rhs = expr._args
        return self.bound(rhs)

    @accumulated
    def visit_inequality(self, expr):
        pass

    @accumulated
    def visit_product(self, expr):
        bounds = [self.bound(c) for c in expr._args]
        return reduce(operator.mul, bounds, 1)

    @accumulated
    def visit_division(self, expr):
        top, bot = expr._args
        return self.bound(top) / self.bound(bot)

    @accumulated
    def visit_linear(self, expr):
        bounds = [expr._coef[id(c)] * self.bound(c) for c in expr._args]
        bounds.append(Bound(expr._const, expr._const))
        return sum(bounds)

    @accumulated
    def visit_sum(self, expr):
        bounds = [self.bound(c) for c in expr._args]
        return sum(bounds)

    @accumulated
    def visit_negation(self, expr):
        assert len(expr._args) == 1

        bound = self.bound(expr._args[0])
        return Bound(-bound.u, -bound.l)

    @accumulated
    def visit_abs(self, expr):
        assert len(expr._args) == 1

        bound = self.bound(expr._args[0])
        upper_bound = max(abs(bound.l), abs(bound.u))
        if bound.l <= 0 and bound.u >= 0:
            return Bound(0, upper_bound)
        else:
            lower_bound = min(abs(bound.l), abs(bound.u))
            return Bound(lower_bound, upper_bound)

    @accumulated
    def visit_pow(self, expr):
        assert len(expr._args) == 2
        base, exponent = expr._args

        warnings.warn('Bounds of pow expression are set [-inf, inf]')
        return Bound(None, None)

    @accumulated
    def visit_unary_function(self, expr):
        assert len(expr._args) == 1

        name = expr._name
        arg = expr._args[0]
        arg_bound = self.bound(arg)

        if name == 'sqrt':
            if arg_bound.l < 0:
                raise DomainError('sqrt')
            return Bound(sqrt(arg_bound.l), sqrt(arg_bound.u))

        elif name == 'log':
            if arg_bound.l <= 0:
                raise DomainError('log')
            return Bound(log(arg_bound.l), log(arg_bound.u))

        elif name == 'asin':
            if arg_bound not in Bound(-1, 1):
                raise DomainError('asin')
            return Bound(asin(arg_bound.l), asin(arg_bound.u))

        elif name == 'acos':
            if arg_bound not in Bound(-1, 1):
                raise DomainError('acos')
            # arccos is a decreasing function, swap upper and lower
            return Bound(acos(arg_bound.u), acos(arg_bound.l))

        elif name == 'atan':
            return Bound(atan(arg_bound.l), atan(arg_bound.u))

        elif name == 'exp':
            return Bound(exp(arg_bound.l), exp(arg_bound.u))

        elif name == 'sin':
            if arg_bound.u - arg_bound.l >= 2 * pi:
                return Bound(-1, 1)
            else:
                return _sin_bound(arg_bound.l, arg_bound.u)

        elif name == 'cos':
            if arg_bound.u - arg_bound.l >= 2 * pi:
                return Bound(-1, 1)
            else:
                # translate left by pi/2
                pi_2 = pi / mpf('2')
                l = arg_bound.l - pi_2
                u = arg_bound.u - pi_2
                # -sin(x - pi/2) == cos(x)
                return Bound(0, 0) - _sin_bound(l, u)

        elif name == 'tan':
            if arg_bound.u - arg_bound.l >= pi:
                return Bound(None, None)
            else:
                l = arg_bound.l % pi
                u = l + (arg_bound.u - arg_bound.l)
                tan_l = tan(l)
                tan_u = tan(u)
                new_l = min(tan_l, tan_u)
                new_u = max(tan_l, tan_u)
                if almosteq(l, 0.5 * pi):
                    new_l = None
                if almosteq(u, 0.5 * pi):
                    new_u = None

                return Bound(new_l, new_u)

        else:
            raise RuntimeError('unknown unary function {}'.format(name))


def _expression_bounds(expr):
    handler = BoundsHandler()
    visit_expression(handler, expr)
    return handler


def expression_bounds(expr):
    """Compute bounds of `expr`."""
    return _expression_bounds(expr).bound(expr)


def is_positive(expr):
    return expression_bounds(expr).is_positive()


def is_nonnegative(expr):
    return expression_bounds(expr).is_nonnegative()


def is_nonpositive(expr):
    return expression_bounds(expr).is_nonpositive()


def is_negative(expr):
    return expression_bounds(expr).is_negative()


def is_zero(expr):
    return expression_bounds(expr).is_positive()

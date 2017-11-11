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
from convexity_detection.expr_visitor import (
    BottomUpExprVisitor,
    ProductExpression,
    DivisionExpression,
    SumExpression,
    LinearExpression,
    NegationExpression,
    AbsExpression,
    UnaryFunctionExpression,
    expr_callback,
)
from convexity_detection.math import *
from convexity_detection.error import DomainError
from pyomo.core.base import _VarData


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


class BoundsVisitor(BottomUpExprVisitor):
    def __init__(self):
        self.memo = {}

    def bound(self, expr):
        if isinstance(expr, Number):
            return Bound(expr, expr)
        else:
            return self.memo[id(expr)]

    def set_bound(self, expr, bound):
        self.memo[id(expr)] = bound

    @expr_callback(_VarData)
    def visit_simple_var(self, v):
        bound = Bound(v.bounds[0], v.bounds[1])
        self.set_bound(v, bound)

    @expr_callback(Number)
    def visit_number(self, n):
        pass  # do nothing

    @expr_callback(ProductExpression)
    def visit_product(self, expr):
        bounds = [self.bound(c) for c in expr._args]
        bound = reduce(operator.mul, bounds, 1)
        self.set_bound(expr, bound)

    @expr_callback(DivisionExpression)
    def visit_division(self, expr):
        top, bot = expr._args
        bound = self.bound(top) / self.bound(bot)
        self.set_bound(expr, bound)

    @expr_callback(LinearExpression)
    def visit_linear(self, expr):
        bounds = [expr._coef[id(c)] * self.bound(c) for c in expr._args]
        bounds.append(Bound(expr._const, expr._const))
        bound = sum(bounds)
        self.set_bound(expr, bound)

    @expr_callback(SumExpression)
    def visit_sum(self, expr):
        bounds = [self.bound(c) for c in expr._args]
        bound = sum(bounds)
        self.set_bound(expr, bound)

    @expr_callback(NegationExpression)
    def visit_negation(self, expr):
        assert len(expr._args) == 1

        bound = self.bound(expr._args[0])
        new_bound = Bound(-bound.u, -bound.l)
        self.set_bound(expr, new_bound)

    @expr_callback(AbsExpression)
    def visit_abs(self, expr):
        assert len(expr._args) == 1

        bound = self.bound(expr._args[0])
        upper_bound = max(abs(bound.l), abs(bound.u))
        if bound.l <= 0 and bound.u >= 0:
            abs_bound = Bound(0, upper_bound)
        else:
            lower_bound = min(abs(bound.l), abs(bound.u))
            abs_bound = Bound(lower_bound, upper_bound)
        self.set_bound(expr, abs_bound)

    @expr_callback(UnaryFunctionExpression)
    def visit_unary_function(self, expr):
        assert len(expr._args) == 1

        name = expr._name
        arg = expr._args[0]
        arg_bound = self.bound(arg)

        if name == 'sqrt':
            if arg_bound.l < 0:
                raise DomainError('sqrt')
            new_bound = Bound(sqrt(arg_bound.l), sqrt(arg_bound.u))

        elif name == 'log':
            if arg_bound.l <= 0:
                raise DomainError('log')
            new_bound = Bound(log(arg_bound.l), log(arg_bound.u))

        elif name == 'asin':
            if arg_bound not in Bound(-1, 1):
                raise DomainError('asin')
            new_bound = Bound(asin(arg_bound.l), asin(arg_bound.u))

        elif name == 'acos':
            if arg_bound not in Bound(-1, 1):
                raise DomainError('acos')
            # arccos is a decreasing function, swap upper and lower
            new_bound = Bound(acos(arg_bound.u), acos(arg_bound.l))

        elif name == 'atan':
            new_bound = Bound(atan(arg_bound.l), atan(arg_bound.u))

        elif name == 'exp':
            new_bound = Bound(exp(arg_bound.l), exp(arg_bound.u))

        elif name == 'sin':
            if arg_bound.u - arg_bound.l >= 2 * pi:
                new_bound = Bound(-1, 1)
            else:
                new_bound = _sin_bound(arg_bound.l, arg_bound.u)

        elif name == 'cos':
            if arg_bound.u - arg_bound.l >= 2 * pi:
                new_bound = Bound(-1, 1)
            else:
                # translate left by pi/2
                pi_2 = pi / mpf('2')
                l = arg_bound.l - pi_2
                u = arg_bound.u - pi_2
                # -sin(x - pi/2) == cos(x)
                new_bound = Bound(0, 0) - _sin_bound(l, u)

        elif name == 'tan':
            if arg_bound.u - arg_bound.l >= pi:
                new_bound = Bound(None, None)
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

                new_bound = Bound(new_l, new_u)

        else:
            raise RuntimeError('unknown unary function {}'.format(name))
        self.set_bound(expr, new_bound)


def expr_bounds(expr):
    """Given an expression, computes its bounds"""
    v = BoundsVisitor()
    v.visit(expr)
    return v.memo[id(expr)]


def _is_positive(bounds, expr):
    bound = bounds[id(expr)]
    return bound.l > 0


def is_positive(expr):
    v = BoundsVisitor()
    v.visit(expr)
    return _is_positive(v.memo, expr)


def _is_nonnegative(bounds, expr):
    bound = bounds[id(expr)]
    return almostgte(bound.l, 0)


def is_nonnegative(expr):
    v = BoundsVisitor()
    v.visit(expr)
    return _is_nonnegative(v.memo, expr)


def _is_nonpositive(bounds, expr):
    bound = bounds[id(expr)]
    return almostlte(bound.u, 0)


def is_nonpositive(expr):
    v = BoundsVisitor()
    v.visit(expr)
    return _is_nonpositive(v.memo, expr)


def _is_negative(bounds, expr):
    bound = bounds[id(expr)]
    return bound.u < 0


def is_negative(expr):
    v = BoundsVisitor()
    v.visit(expr)
    return _is_negative(v.memo, expr)


def _is_zero(bounds, expr):
    bound = bounds[id(expr)]
    return almosteq(bound.l, 0) and almosteq(bound.u, 0)

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
from enum import Enum
from collections import defaultdict
import pyomo.environ as aml
from convexity_detection.bounds import (
    BoundsVisitor,
    _is_nonnegative,
    is_nonnegative,
    _is_nonpositive,
    is_nonpositive,
    _is_negative,
    _is_positive,
    _is_zero,
)
from convexity_detection.linearity import (
    MonotonicityExprVisitor,
    Monotonicity,
)
from convexity_detection.expr_visitor import (
    ExprVisitor,
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
from pyomo.core.base.var import SimpleVar


class Convexity(Enum):
    Convex = 0
    Concave = 1
    Linear = 2
    Unknown = 3

    def is_convex(self):
        return self == self.Convex or self == self.Linear

    def is_concave(self):
        return self == self.Concave or self == self.Linear

    def is_linear(self):
        return self == self.Linear

    def is_unknown(self):
        return self == self.Unknown


def _sin_convexity(bound, cvx, arg):
    # if bound is like [0, pi], we need to be extra carefull
    diff = bound.u - bound.l
    if diff > pi:
        return Convexity.Unknown

    sin_l = sin(bound.l)
    sin_u = sin(bound.u)
    if sin_l * sin_u < 0 and not almosteq(sin_l*sin_u, 0):
        return Convexity.Unknown

    l = bound.l % (2 * pi)
    u = l + diff
    if l <= 0.5 * pi <= u:
        cond_linear = cvx.is_linear()
        cond_convex = (
            cvx.is_convex() and is_nonpositive(aml.cos(arg))
        )
        cond_concave = (
            cvx.is_concave() and is_nonnegative(aml.cos(arg))
        )
        if cond_linear or cond_convex or cond_concave:
            return Convexity.Concave
    elif l <= 1.5 * pi <= u:
        cond_linear = cvx.is_linear()
        cond_concave = (
            cvx.is_concave() and is_nonpositive(aml.cos(arg))
        )
        cond_convex = (
            cvx.is_convex() and is_nonnegative(aml.cos(arg))
        )
        if cond_linear or cond_concave or cond_convex:
            return Convexity.Convex
    return Convexity.Uknown


def _cos_convexity(bound, cvx, arg):
    # if bound is like [0, pi], we need to be extra carefull
    diff = bound.u - bound.l
    if diff > pi:
        return Convexity.Unknown

    cos_l = cos(bound.l)
    cos_u = cos(bound.u)
    if cos_l * cos_u < 0 and not almosteq(cos_l*cos_u, 0):
        return Convexity.Unknown

    l = bound.l % (2 * pi)
    u = l + diff
    if u <= 0.5*pi or l >= 1.5*pi:
        cond_linear = cvx.is_linear()
        cond_convex = (
            cvx.is_convex() and is_nonpositive(aml.sin(arg))
        )
        cond_concave = (
            cvx.is_concave() and is_nonnegative(aml.sin(arg))
        )
        if cond_linear or cond_convex or cond_concave:
            return Convexity.Concave
    else:
        cond_linear = cvx.is_linear()
        cond_concave = (
            cvx.is_concave() and is_nonpositive(aml.sin(arg))
        )
        cond_convex = (
            cvx.is_convex() and is_nonnegative(aml.sin(arg))
        )
        if cond_linear or cond_concave or cond_convex:
            return Convexity.Convex
    return Convexity.Uknown


def _tan_convexity(bound, cvx):
    diff = bound.u - bound.l
    if diff > 0.5 * pi:
        return Convexity.Unknown

    tan_l = tan(bound.l)
    tan_u = tan(bound.u)
    if tan_l * tan_u < 0 and not almosteq(tan_l*tan_u, 0):
        return Convexity.Unknown

    if tan_u >= 0 and tan_l >= 0 and cvx.is_convex():
        return Convexity.Convex
    if tan_u <= 0 and tan_l <= 0 and cvx.is_concave():
        return Convexity.Concave
    return Convexity.Unknown


def _asin_convexity(bound, cvx):
    if -1 <= bound.l <= bound.u <= 0 and cvx.is_concave():
        return Convexity.Concave

    if 0 <= bound.l <= bound.u <= 1 and cvx.is_convex():
        return Convexity.Convex

    return Convexity.Unknown


def _acos_convexity(bound, cvx):
    if -1 <= bound.l <= bound.u <= 0 and cvx.is_concave():
        return Convexity.Convex

    if 0 <= bound.l <= bound.u <= 1 and cvx.is_convex():
        return Convexity.Concave

    return Convexity.Unknown


def _atan_convexity(bound, cvx):
    if bound.u <= 0 and cvx.is_convex():
        return Convexity.Convex

    if bound.l >= 0 and cvx.is_concave():
        return Convexity.Concave

    return Convexity.Unknown


class ConvexityExprVisitor(BottomUpExprVisitor):
    def __init__(self):
        self.bounds = None
        self.mono = None
        self.memo = {}

    def visit(self, expr):
        mono_visitor = MonotonicityExprVisitor()
        mono_visitor.visit(expr)
        self.mono = mono_visitor.memo
        self.bounds = mono_visitor.bounds
        return super().visit(expr)

    def convexity(self, expr):
        if isinstance(expr, Number):
            return Convexity.Linear
        else:
            return self.memo[id(expr)]

    def set_convexity(self, expr, cvx):
        self.memo[id(expr)] = cvx

    def monotonicity(self, expr):
        if isinstance(expr, Number):
            return Monotonicity.Constant
        else:
            return self.mono[id(expr)]

    def is_nonnegative(self, expr):
        if isinstance(expr, Number):
            return expr >= 0
        else:
            return _is_nonnegative(self.bounds, expr)

    def is_nonpositive(self, expr):
        if isinstance(expr, Number):
            return expr <= 0
        else:
            return _is_nonpositive(self.bounds, expr)

    def is_negative(self, expr):
        if isinstance(expr, Number):
            return expr < 0
        else:
            return _is_negative(self.bounds, expr)

    def is_positive(self, expr):
        if isinstance(expr, Number):
            return expr > 0
        else:
            return _is_positive(self.bounds, expr)

    def is_zero(self, expr):
        if isinstance(expr, Number):
            return almosteq(expr, 0)
        else:
            return _is_zero(self.bounds, expr)

    def bound(self, expr):
        if isinstance(expr, Number):
            return Bound(expr, expr)
        else:
            return self.bounds[id(expr)]

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        self.set_convexity(v, Convexity.Linear)

    @expr_callback(Number)
    def visit_number(self, n):
        pass

    @expr_callback(SumExpression)
    def visit_sum(self, expr):
        self.visit_linear(expr)

    @expr_callback(LinearExpression)
    def visit_linear(self, expr):
        def _adjust_convexity(cvx, coef):
            if cvx.is_unknown() or cvx.is_linear():
                return cvx

            if coef > 0:
                return cvx
            elif coef < 0:
                if cvx.is_convex():
                    return Convexity.Concave
                else:
                    return Convexity.Convex
            else:
                # if coef == 0, it's a constant with value 0.0
                return Convexity.Linear

        if hasattr(expr, '_coef'):
            coefs = expr._coef
        else:
            coefs = defaultdict(lambda: 1.0)

        cvxs = [
            _adjust_convexity(self.convexity(a), coefs[id(a)])
            for a in expr._args
        ]
        all_linear = all([c.is_linear() for c in cvxs])
        all_convex = all([c.is_convex() for c in cvxs])
        all_concave = all([c.is_concave() for c in cvxs])

        # is it true that if all args are linear then the expr is linear?
        if all_linear:
            self.set_convexity(expr, Convexity.Linear)
        elif all_convex:
            self.set_convexity(expr, Convexity.Convex)
        elif all_concave:
            self.set_convexity(expr, Convexity.Concave)
        else:
            self.set_convexity(expr, Convexity.Unknown)

    @expr_callback(AbsExpression)
    def visit_abs(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        cvx = self.convexity(arg)

        if cvx.is_linear():
            self.set_convexity(expr, Convexity.Convex)
        elif cvx.is_convex():
            if self.is_zero(arg):
                self.set_convexity(expr, Convexity.Linear)
            elif self.is_nonnegative(arg):
                self.set_convexity(expr, Convexity.Convex)
            elif self.is_nonpositive(arg):
                self.set_convexity(expr, Convexity.Concave)
            else:
                self.set_convexity(expr, Convexity.Unknown)
        elif cvx.is_concave():
            if self.is_zero(arg):
                self.set_convexity(expr, Convexity.Linear)
            elif self.is_nonnegative(arg):
                self.set_convexity(expr, Convexity.Concave)
            elif self.is_nonpositive(arg):
                self.set_convexity(expr, Convexity.Convex)
            else:
                self.set_convexity(expr, Convexity.Unknown)
        else:
            self.set_convexity(expr, Convexity.Unknown)

    @expr_callback(NegationExpression)
    def visit_negation(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        cvx = self.convexity(arg)

        if cvx.is_linear():
            self.set_convexity(expr, Convexity.Linear)
        elif cvx.is_convex():
            self.set_convexity(expr, Convexity.Concave)
        elif cvx.is_concave():
            self.set_convexity(expr, Convexity.Convex)
        else:
            self.set_convexity(expr, Convexity.Unknown)

    @expr_callback(UnaryFunctionExpression)
    def visit_unary_function(self, expr):
        assert len(expr._args) == 1
        name = expr._name
        arg = expr._args[0]
        bound = self.bound(arg)
        cvx = self.convexity(arg)

        if name == 'sqrt':
            # TODO: handle sqrt(x*x) which is convex
            if self.is_nonnegative(arg) and cvx.is_concave():
                self.set_convexity(expr, Convexity.Concave)
            else:
                self.set_convexity(expr, Convexity.Unknown)

        elif name == 'exp':
            if cvx.is_convex():
                self.set_convexity(expr, Convexity.Convex)
            else:
                self.set_convexity(expr, Convexity.Unknown)

        elif name == 'log':
            # TODO: handle log(exp(x)) == x
            if self.is_positive(arg) and cvx.is_concave():
                self.set_convexity(expr, Convexity.Concave)
            else:
                self.set_convexity(expr, Convexity.Unknown)

        elif name == 'sin':
            self.set_convexity(expr, _sin_convexity(bound, cvx, arg))

        elif name == 'cos':
            self.set_convexity(expr, _cos_convexity(bound, cvx, arg))

        elif name == 'tan':
            self.set_convexity(expr, _tan_convexity(bound, cvx))

        elif name == 'asin':
            self.set_convexity(expr, _asin_convexity(bound, cvx))

        elif name == 'acos':
            self.set_convexity(expr, _acos_convexity(bound, cvx))

        elif name == 'atan':
            self.set_convexity(expr, _atan_convexity(bound, cvx))

        else:
            raise RuntimeError('unknown unary function {}'.format(name))


def expr_convexity(expr):
    visitor = ConvexityExprVisitor()
    visitor.visit(expr)
    return visitor.convexity(expr)

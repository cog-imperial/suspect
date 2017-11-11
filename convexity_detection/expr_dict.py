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
from functools import reduce
from convexity_detection.expr_visitor import (
    BottomUpExprVisitor,
    expr_callback,
    ProductExpression,
    LinearExpression,
    EqualityExpression,
    InequalityExpression,
    DivisionExpression,
    SumExpression,
    UnaryFunctionExpression,
    AbsExpression,
)
from convexity_detection.math import almosteq
from pyomo.core.base import (
    _VarData,
    NumericConstant,
)
from pyomo.core.base.constraint import _GeneralConstraintData


UNHASHABLE_TYPES = (Number, NumericConstant)


class ExprHasher(BottomUpExprVisitor):
    def __init__(self, float_hasher=None):
        self.memo = {}
        self._float_hasher = float_hasher

    def hash(self, expr):
        if isinstance(expr, UNHASHABLE_TYPES):
            if self._float_hasher is not None:
                return self._float_hash(expr)
            else:
                return 0
        else:
            return self.memo[id(expr)]

    def set_hash(self, expr, h):
        self.memo[id(expr)] = h

    def _float_hash(self, expr):
        if isinstance(expr, Number):
            return self._float_hasher.hash(expr)
        elif isinstance(expr, NumericConstant):
            return self._float_hasher.hash(expr.value)
        else:
            raise RuntimeError('unknown float {} of type {}'.format(
                str(expr), type(expr)
            ))

    def _type_xor_args(self, expr):
        hashes = [self.hash(e) for e in expr._args]
        return hash(type(expr)) ^ reduce(lambda x, y: x ^ y, hashes, 0)

    @expr_callback((Number, NumericConstant))
    def visit_number(self, n):
        pass

    @expr_callback(_VarData)
    def visit_var_data(self, v):
        self.set_hash(v, hash(_VarData) ^ id(v))

    @expr_callback(InequalityExpression)
    def visit_inequality(self, expr):
        # TODO: handle a <= b <= c etc.
        # not associative
        h = self._type_xor_args(expr)
        self.set_hash(expr, h)

    @expr_callback(UnaryFunctionExpression)
    def visit_unary_function(self, expr):
        assert len(expr._args) == 1
        h = self._type_xor_args(expr) ^ hash(expr.name)
        self.set_hash(expr, h)

    @expr_callback(LinearExpression)
    def visit_linear_expr(self, expr):
        coef = expr._coef
        hashes = [
            self.hash(e) ^ self.hash(coef[id(e)])
            for e in expr._args
        ]
        h = hash(type(expr)) ^ reduce(lambda x, y: x ^ y, hashes, 0)
        self.set_hash(expr, h)

    @expr_callback(object)
    def visit_expr(self, expr):
        h = self._type_xor_args(expr)
        self.set_hash(expr, h)


def expr_hash(expr, float_hasher=None):
    if isinstance(expr, _GeneralConstraintData):
        expr = expr.expr
    h = ExprHasher(float_hasher=float_hasher)
    h.visit(expr)
    return h.memo[id(expr)]


def _is_leaf_node(expr):
    return isinstance(expr, (Number, NumericConstant, _VarData))


def expr_equal(expr1, expr2):
    stack = [(expr1, expr2)]
    while len(stack) > 0:
        e1, e2 = stack.pop()

        if type(e1) != type(e2):
            return False

        if _is_leaf_node(e1):
            if isinstance(e1, Number):
                if not almosteq(e1, e2):
                    return False

            if isinstance(e1, NumericConstant):
                if not almosteq(e1.value, e2.value):
                    return False

            if isinstance(e1, _VarData):
                if id(e1) != id(e2):
                    return False
        else:
            if len(e1._args) != len(e2._args):
                return False

            if isinstance(e1, LinearExpression):
                for expr_id, c in e1._coef.items():
                    if expr_id not in e2._coef:
                        return False

                    if not almosteq(c, e2._coef[expr_id]):
                        return False

            if isinstance(e1, UnaryFunctionExpression):
                if e1.name != e2.name:
                    return False

            # all checks passed, check args
            for a1, a2 in zip(e1._args, e2._args):
                stack.append((a1, a2))

    return True


class ExpressionDict(object):
    def __init__(self, float_hasher=None):
        self._float_hasher = float_hasher
        self._data = {}

    def _hash(self, expr):
        hasher = ExprHasher(float_hasher=self._float_hasher)
        hasher.visit(expr)
        return hasher.memo[id(expr)]

    def set(self, expr, v):
        h = self._hash(expr)
        if h not in self._data:
            self._data[h] = [(expr, v)]
        else:
            data = self._data[h]
            for i in range(len(data)):
                old_exp, old_value = data[i]
                if expr_equal(old_exp, expr):
                    data[i] = (expr, v)
                    return
            # no matching expr found, just append
            data.append((expr, v))

    def get(self, expr):
        h = self._hash(expr)
        if h not in self._data:
            return None
        data = self._data[h]
        for (target_expr, value) in data:
            if expr_equal(target_expr, expr):
                return value
        return None

    def __len__(self):
        length = 0
        for _, values in self._data.items():
            length += len(values)
        return length

    def __getitem__(self, expr):
        return self.get(expr)

    def __setitem__(self, expr, value):
        return self.set(expr, value)

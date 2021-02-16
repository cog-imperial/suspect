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
from suspect.math import almosteq
from suspect.float_hash import RoundFloatHasher
from pyomo.core.expr.numeric_expr import (
    nonpyomo_leaf_types,
    LinearExpression,
    UnaryFunctionExpression,
)
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.expr.visitor import ExpressionValueVisitor
from pyomo.core.base import _VarData
from pyomo.core.base.constraint import _GeneralConstraintData


class ExpressionHasherVisitor(ExpressionValueVisitor):
    def __init__(self, float_hasher=None):
        if float_hasher is None:
            float_hasher = RoundFloatHasher()
        self._float_hasher = float_hasher

    def hash(self, expr):
        if expr.__class__ in nonpyomo_leaf_types:
            return self._float_hasher.hash(float(expr))
        return self.memo[id(expr)]

    def set_hash(self, expr, h):
        self.memo[id(expr)] = h

    def _type_xor_args(self, expr, values):
        return hash(type(expr)) ^ reduce(lambda x, y: x ^ y, values, 0)

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            return True, self._float_hasher.hash(float(node))

        if node.is_constant():
            return True, self._float_hasher.hash(float(node.value))

        if node.is_variable_type():
            h = hash(_VarData) ^ id(node)
            return True, h

        return False, None

    def visit(self, node, values):
        return self._type_xor_args(node, values)

    def visit_linear(self, expr):
        coef = expr._coef
        hashes = [
            self.hash(e) ^ self.hash(coef[id(e)])
            for e in expr._args
        ]
        h = hash(type(expr)) ^ reduce(lambda x, y: x ^ y, hashes, 0)
        self.set_hash(expr, h)


def expr_hash(expr, float_hasher=None):
    if isinstance(expr, _GeneralConstraintData):
        expr = expr.expr

    hasher = ExpressionHasherVisitor(float_hasher=float_hasher)
    return hasher.dfs_postorder_stack(expr)


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
            if len(e1._args_) != len(e2._args_):
                return False

            if isinstance(e1, LinearExpression):
                if len(e1.linear_vars) != len(e2.linear_vars):
                    return False

                for v1, v2 in zip(e1.linear_vars, e2.linear_vars):
                    if id(v1) != id(v2):
                        return False

                for c1, c2 in zip(e2.linear_coefs, e2.linear_coefs):
                    if not almosteq(c1, c2):
                        return False

                if not almosteq(e1.constant, e2.constant):
                    return False

            if isinstance(e1, UnaryFunctionExpression):
                if e1.name != e2.name:
                    return False

            # all checks passed, check args
            for a1, a2 in zip(e1._args_, e2._args_):
                stack.append((a1, a2))

    return True


class ExpressionDict(object):
    def __init__(self, float_hasher=None):
        self._float_hasher = float_hasher
        self._data = {}

    def _hash(self, expr):
        return expr_hash(expr)

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

    def _dump(self):
        print('Begin ExpressionDict Dump')
        for _, values in self._data.items():
            for expr, v in values:
                print('[{}] -> {}'.format(expr, v))
        print('End')


class TightestExpressionDict(ExpressionDict):
    """Like `ExpressionDict`, but when setting the bounds it will tighten them."""
    def tighten(self, expr, value):
        if value is not None:
            old_bound = self.get(expr)
            if old_bound is None:
                new_bound = value
            else:
                new_bound = old_bound.tighten(value)
            self.set(expr, new_bound)

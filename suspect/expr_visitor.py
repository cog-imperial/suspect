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

from abc import ABC, abstractmethod
from numbers import Number
from pyomo.core.base import Var
from pyomo.core.base.var import (_GeneralVarData, SimpleVar)
import pyomo.core.base.expr_pyomo4 as omo
import functools


Variable = Var
NumericConstant = omo.NumericConstant
EqualityExpression = omo._EqualityExpression
InequalityExpression = omo._InequalityExpression
ProductExpression = omo._ProductExpression
DivisionExpression = omo._DivisionExpression
SumExpression = omo._SumExpression
LinearExpression = omo._LinearExpression
NegationExpression = omo._NegationExpression
UnaryFunctionExpression = omo._UnaryFunctionExpression
AbsExpression = omo._AbsExpression
PowExpression = omo._PowExpression


CALLBACKS_NAMES = {
    _GeneralVarData: 'visit_variable',
    SimpleVar: 'visit_variable',
    Variable: 'visit_variable',
    int: 'visit_number',
    float: 'visit_number',
    Number: 'visit_number',
    NumericConstant: 'visit_numeric_constant',
    EqualityExpression: 'visit_equality',
    InequalityExpression: 'visit_inequality',
    ProductExpression: 'visit_product',
    DivisionExpression: 'visit_division',
    SumExpression: 'visit_sum',
    LinearExpression: 'visit_linear',
    NegationExpression: 'visit_negation',
    UnaryFunctionExpression: 'visit_unary_function',
    AbsExpression: 'visit_abs',
    PowExpression: 'visit_pow',

}


def try_callback(handler, expr):
    """Try calling one of the registered callbacks, raising an exception
    if no callback matches.
    """
    callback_name = CALLBACKS_NAMES.get(type(expr))
    if callback_name is None:
        msg = 'No callback found for {} {}'.format(
            expr, type(expr)
        )
        raise RuntimeError(msg)
    else:
        callback = getattr(handler, callback_name)
        if callback is None:
            raise RuntimeError('Missing callback {}'.format(callback_name))
        callback(expr)


class accumulated(object):
    def __init__(self, fun):
        self._fun = fun

    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

    def __call__(self, obj, expr):
        value = self._fun(obj, expr)
        obj.accumulate(expr, value)


class visit_after(object):
    def __init__(self, other):
        self.other_name = other

    def __call__(self, fun):
        fun_name = fun.__name__

        def new_fun(obj, *args, **kwargs):
            other_handler = getattr(obj, self.other_name)
            other_fun = getattr(other_handler, fun_name)
            other_fun(*args, **kwargs)
            return fun(obj, *args, **kwargs)
        return new_fun


class ExpressionHandler(ABC):
    @abstractmethod
    def visit_number(self, n):
        pass

    @abstractmethod
    def visit_numeric_constant(self, expr):
        pass

    @abstractmethod
    def visit_variable(self, expr):
        pass

    @abstractmethod
    def visit_equality(self, expr):
        pass

    @abstractmethod
    def visit_inequality(self, expr):
        pass

    @abstractmethod
    def visit_product(self, expr):
        pass

    @abstractmethod
    def visit_division(self, expr):
        pass

    @abstractmethod
    def visit_sum(self, expr):
        pass

    @abstractmethod
    def visit_linear(self, expr):
        pass

    @abstractmethod
    def visit_negation(self, expr):
        pass

    @abstractmethod
    def visit_unary_function(self, expr):
        pass

    @abstractmethod
    def visit_abs(self, expr):
        pass

    @abstractmethod
    def visit_pow(self, expr):
        pass


def visit(handler, expr):
    """Visit the expression from root to leaves.

    Parameters
    ----------
    handler: ExpressionHandler
        handler that will handle each sub-expression callback
    expr:
        the expression to visit
    """
    with omo.bypass_clone_check():
        stack = [(0, expr)]
        while len(stack) > 0:
            (lvl, expr) = stack.pop()
            if isinstance(expr, omo._ExpressionBase):
                for arg in expr._args:
                    stack.append((lvl+1, arg))

            try_callback(handler, expr)


def bottom_up_visit(handler, expr):
    """Visit the expression from leaf nodes to root.

    Parameters
    ----------
    handler: ExpressionHandler
        handler that will handle each sub-expression callback
    expr:
        the expression to visit
    """
    expr_level = {}
    expr_by_id = {}
    with omo.bypass_clone_check():
        stack = [(0, expr)]
        while len(stack) > 0:
            (lvl, expr) = stack.pop()

            old_lvl = expr_level.get(id(expr), -1)
            expr_level[id(expr)] = max(old_lvl, lvl)
            expr_by_id[id(expr)] = expr

            if isinstance(expr, omo._ExpressionBase):
                for arg in expr._args:
                    stack.append((lvl+1, arg))

        expr_level = sorted(
            [(lvl, ex) for ex, lvl in expr_level.items()],
            reverse=True,
        )

        for _, expr_id in expr_level:
            expr = expr_by_id[expr_id]
            try_callback(handler, expr)
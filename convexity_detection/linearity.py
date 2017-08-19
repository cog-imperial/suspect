import numpy as np
from numbers import Number
import operator
from functools import reduce
from convexity_detection.expr_visitor import (
    ExprVisitor,
    BottomUpExprVisitor,
    ProductExpression,
    SumExpression,
    LinearExpression,
    expr_callback,
)
from pyomo.core.base.var import SimpleVar


class IsConstantVisitor(ExprVisitor):
    def __init__(self):
        self.variable_found = False

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        if not v.fixed:
            self.variable_found = True

    @expr_callback(object)
    def visit_any(self, o):
        pass


def is_constant(expr):
    visitor = IsConstantVisitor()
    visitor.visit(expr)
    return not visitor.variable_found


class IsLinearVisitor(BottomUpExprVisitor):
    def __init__(self):
        self.memo = {}

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        self.memo[id(v)] = True

    @expr_callback(LinearExpression)
    def visit_linear(self, expr):
        self.memo[id(expr)] = all(self.memo[id(a)] for a in expr._args)

    @expr_callback(SumExpression)
    def visit_sum(self, expr):
        self.memo[id(expr)] = all(self.memo[id(a)] for a in expr._args)

    @expr_callback(ProductExpression)
    def visit_product(self, expr):
        # only one arg gets to be non-constant
        constants = [a for a in expr._args if is_constant(a)]
        self.memo[id(expr)] = len(constants) >= len(expr._args) - 1


def is_linear(expr):
    visitor = IsLinearVisitor()
    visitor.visit(expr)
    return visitor.memo[id(expr)]


def is_quadratic(expr):
    raise NotImplementedError('is_quadratic')


def is_polynomial(expr):
    raise NotImplementedError('is_polynomial')


class IsNondecreasingExprVisitor(BottomUpExprVisitor):
    def __init__(self):
        self.memo = {}

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        self.memo[id(v)] = True

    @expr_callback(LinearExpression)
    def visit_linear(self, expr):
        self.memo[id(expr)] = all(self.memo[id(a)] for a in expr._args)



def is_nondecreasing(expr):
    visitor = IsNondecreasingExprVisitor()
    visitor.visit(expr)
    return visitor.memo[id(expr)]

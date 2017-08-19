import numpy as np
from numbers import Number
import operator
from functools import reduce
from convexity_detection.expr_visitor import (
    BottomUpExprVisitor,
    ProductExpression,
    SumExpression,
    LinearExpression,
    expr_callback,
)
from pyomo.core.base.var import SimpleVar


class Bound(object):
    def __init__(self, l, u):
        if l is None:
            l = -np.inf
        if u is None:
            u = np.inf
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
                return Bound(-np.inf, np.inf)
            else:
                return self.__mul__(Bound(1/ou, 1/ol))
        elif isinstance(other, Number):
            return self.__truediv__(Bound(other, other))
        else:
            raise TypeError('dividing Bound by incompatible type')

    def __eq__(self, other):
        if not isinstance(other, Bound):
            return False
        return np.isclose(self.l, other.l) and np.isclose(self.u, other.u)

    def __repr__(self):
        return '<{} at {}>'.format(str(self), id(self))

    def __str__(self):
        return '[{}, {}]'.format(self.l, self.u)


class BoundsVisitor(BottomUpExprVisitor):
    def __init__(self):
        self.memo = {}

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        bound = Bound(v.bounds[0], v.bounds[1])
        self.memo[id(v)] = bound

    @expr_callback(ProductExpression)
    def visit_product(self, expr):
        bounds = [self.memo[id(c)] for c in expr._args]
        bound = reduce(operator.mul, bounds, 1)
        self.memo[id(expr)] = bound

    @expr_callback(LinearExpression)
    def visit_linear(self, expr):
        bounds = [expr._coef[id(c)] * self.memo[id(c)] for c in expr._args]
        bound = sum(bounds)
        self.memo[id(expr)] = bound

    @expr_callback(SumExpression)
    def visit_sum(self, expr):
        bounds = [self.memo[id(c)] for c in expr._args]
        bound = sum(bounds)
        self.memo[id(expr)] = bound


def expr_bounds(expr):
    """Given an expression, computes its bounds"""
    v = BoundsVisitor()
    v.visit(expr)
    return v.memo[id(expr)]


if __name__ == '__main__':
    import pyomo.environ as aml
    from pyomo_compat import set_pyomo4_expression_tree
    set_pyomo4_expression_tree()
    x = aml.Var()
    y = aml.Var()
    z = aml.Var()
    expr0 = (x + y) * z

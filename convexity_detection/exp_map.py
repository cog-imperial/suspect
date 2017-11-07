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


class ExpressionMap(object):
    def __init__(self):
        pass

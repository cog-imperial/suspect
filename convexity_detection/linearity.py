from numbers import Number
from enum import Enum
from collections import defaultdict
import pyomo.environ as aml
from convexity_detection.bounds import (
    BoundsVisitor,
    _is_nonnegative,
    _is_nonpositive,
    is_nonnegative,
    is_nonpositive,
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


class Monotonicity(Enum):
    Nondecreasing = 0
    Nonincreasing = 1
    Constant = 2
    Unknown = 3

    def is_nondecreasing(self):
        return self == self.Nondecreasing or self == self.Constant

    def is_nonincreasing(self):
        return self == self.Nonincreasing or self == self.Constant

    def is_constant(self):
        return self == self.Constant

    def is_unknown(self):
        return self == self.Unknown


class MonotonicityExprVisitor(BottomUpExprVisitor):
    def __init__(self):
        self.bounds = None
        self.memo = {}

    def visit(self, expr):
        bounds_visitor = BoundsVisitor()
        bounds_visitor.visit(expr)
        self.bounds = bounds_visitor.memo
        return super().visit(expr)

    def monotonicity(self, expr):
        if isinstance(expr, Number):
            return Monotonicity.Constant
        else:
            return self.memo[id(expr)]

    def set_monotonicity(self, expr, mono):
        self.memo[id(expr)] = mono

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

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        self.set_monotonicity(v, Monotonicity.Nondecreasing)

    @expr_callback(Number)
    def visit_number(self, n):
        pass

    @expr_callback(SumExpression)
    def visit_sum(self, expr):
        self.visit_linear(expr)

    @expr_callback(LinearExpression)
    def visit_linear(self, expr):
        def _adjust_monotonicity(mono, coef):
            if mono.is_unknown() or mono.is_constant():
                return mono

            if coef > 0:
                return mono
            elif coef < 0:
                if mono.is_nondecreasing():
                    return Monotonicity.Nonincreasing
                else:
                    return Monotonicity.Nondecreasing
            else:
                # if coef == 0, it's a constant with value 0.0
                return Monotonicity.Constant

        if hasattr(expr, '_coef'):
            coefs = expr._coef
        else:
            coefs = defaultdict(lambda: 1.0)

        monos = [
            _adjust_monotonicity(self.monotonicity(a), coefs[id(a)])
            for a in expr._args
        ]
        all_nondec = all([m.is_nondecreasing() for m in monos])
        all_noninc = all([m.is_nonincreasing() for m in monos])

        if all_nondec:
            self.set_monotonicity(expr, Monotonicity.Nondecreasing)
        elif all_noninc:
            self.set_monotonicity(expr, Monotonicity.Nonincreasing)
        else:
            self.set_monotonicity(expr, Monotonicity.Unknown)

    @expr_callback(ProductExpression)
    def visit_product(self, expr):
        assert len(expr._args) == 2

        def _product_mono(f, g):
            # Rules taken from Appendix
            mono_f = self.monotonicity(f)
            mono_g = self.monotonicity(g)

            if mono_f.is_constant() and mono_g.is_constant():
                return Monotonicity.Constant
            elif mono_g.is_constant():
                if mono_f.is_nondecreasing() and self.is_nonnegative(g):
                    return Monotonicity.Nondecreasing
                elif mono_f.is_nonincreasing() and self.is_nonpositve(g):
                    return Monotonicity.Nondecreasing
                elif mono_f.is_nondecreasing() and self.is_nonpositive(g):
                    return Monotonicity.Nonincreasing
                elif mono_f.is_nonincreasing() and self.is_nonnegative(g):
                    return Monotonicity.Nonincreasing
                else:
                    return Monotonicity.Unknown
            elif mono_f.is_constant():
                return _product_mono(g, f)
            else:
                nondec_cond1 = (
                    mono_f.is_nondecreasing() and self.is_nonnegative(g)
                ) or (
                    mono_f.is_nonincreasing() and self.is_nonpositive(g)
                )
                nondec_cond2 = (
                    self.is_nonnegative(f) and mono_g.is_nondecreasing()
                ) or (
                    self.is_nonpositive(f) and mono_g.is_nonincreasing()
                )
                noninc_cond1 = (
                    mono_f.is_nonincreasing() and self.is_nonnegative(g)
                ) or (
                    mono_f.is_nondecreasing() and self.is_nonpositive(g)
                )
                noninc_cond2 = (
                    self.is_nonnegative(f) and mono_g.is_nonincreasing()
                ) or (
                    self.is_nonpositive(f) and mono_g.is_nondecreasing()
                )
                if nondec_cond1 and nondec_cond2:
                    return Monotonicity.Nondecreasing
                elif noninc_cond1 and noninc_cond2:
                    return Monotonicity.Nonincreasing
                else:
                    return Monotonicity.Unknown

        f, g = expr._args
        mono = _product_mono(f, g)
        self.set_monotonicity(expr, mono)

    @expr_callback(DivisionExpression)
    def visit_division(self, expr):
        assert len(expr._args) == 2

        def _division_mono(f, g):
            # Rules taken from Appendix
            # Notice that it's very similar to product, with the difference
            # being in the quotient
            mono_f = self.monotonicity(f)
            mono_g = self.monotonicity(g)

            if mono_f.is_constant() and mono_g.is_constant():
                return Monotonicity.Constant
            elif mono_g.is_constant():
                if mono_f.is_nondecreasing() and self.is_nonnegative(g):
                    return Monotonicity.Nondecreasing
                elif mono_f.is_nonincreasing() and self.is_nonpositve(g):
                    return Monotonicity.Nondecreasing
                elif mono_f.is_nondecreasing() and self.is_nonpositive(g):
                    return Monotonicity.Nonincreasing
                elif mono_f.is_nonincreasing() and self.is_nonnegative(g):
                    return Monotonicity.Nonincreasing
                else:
                    return Monotonicity.Unknown
            else:
                nondec_cond1 = (
                    mono_f.is_nondecreasing() and self.is_nonnegative(g)
                ) or (
                    mono_f.is_nonincreasing() and self.is_nonpositive(g)
                )
                nondec_cond2 = (
                    self.is_nonnegative(f) and mono_g.is_nonincreasing()
                ) or (
                    self.is_nonpositive(f) and mono_g.is_nondecreasing()
                )
                noninc_cond1 = (
                    mono_f.is_nonincreasing() and self.is_nonnegative(g)
                ) or (
                    mono_f.is_nondecreasing() and self.is_nonpositive(g)
                )
                noninc_cond2 = (
                    self.is_nonnegative(f) and mono_g.is_nondecreasing()
                ) or (
                    self.is_nonpositive(f) and mono_g.is_nonincreasing()
                )
                if nondec_cond1 and nondec_cond2:
                    return Monotonicity.Nondecreasing
                elif noninc_cond1 and noninc_cond2:
                    return Monotonicity.Nonincreasing
                else:
                    return Monotonicity.Unknown

        f, g = expr._args
        mono = _division_mono(f, g)
        self.set_monotonicity(expr, mono)

    @expr_callback(AbsExpression)
    def visit_abs(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        mono = self.monotonicity(arg)

        # good examples to understand the behaviour of abs are abs(-x) and
        # abs(1/x)
        if self.is_nonnegative(arg):
            # abs(x), x > 0 is the same as x
            self.set_monotonicity(expr, mono)
        elif self.is_nonpositive(arg):
            # abs(x), x < 0 is the opposite of x
            if mono.is_nondecreasing():
                self.set_monotonicity(expr, Monotonicity.Nonincreasing)
            else:
                self.set_monotonicity(expr, Monotonicity.Nondecreasing)
        else:
            self.set_monotonicity(expr, Monotonicity.Unknown)

    @expr_callback(NegationExpression)
    def visit_negation(self, expr):
        assert len(expr._args) == 1
        arg = expr._args[0]
        mono = self.monotonicity(arg)

        if mono.is_nondecreasing():
            self.set_monotonicity(expr, Monotonicity.Nonincreasing)
        elif mono.is_nonincreasing():
            self.set_monotonicity(expr, Monotonicity.Nondecreasing)
        else:
            self.set_monotonicity(expr, Monotonicity.Unknown)

    @expr_callback(UnaryFunctionExpression)
    def visit_unary_function(self, expr):
        assert len(expr._args) == 1

        name = expr._name
        arg = expr._args[0]
        mono = self.monotonicity(arg)

        if name in ['sqrt', 'log', 'asin', 'atan', 'tan', 'exp']:
            self.set_monotonicity(expr, mono)

        elif name in ['acos']:
            if mono.is_constant():
                self.set_monotonicity(expr, Monotonicity.Constant)
            elif mono.is_nondecreasing():
                self.set_monotonicity(expr, Monotonicity.Nonincreasing)
            elif mono.is_nonincreasing():
                self.set_monotonicity(expr, Monotonicity.Nondecreasing)
            else:
                self.set_monotonicity(expr, Monotonicity.Unknown)

        elif name == 'sin':
            cos_arg = aml.cos(arg)
            if mono.is_nondecreasing() and is_nonnegative(cos_arg):
                self.set_monotonicity(expr, Monotonicity.Nondecreasing)
            elif mono.is_nonincreasing() and is_nonpositive(cos_arg):
                self.set_monotonicity(expr, Monotonicity.Nondecreasing)
            elif mono.is_nonincreasing() and is_nonnegative(cos_arg):
                self.set_monotonicity(expr, Monotonicity.Nonincreasing)
            elif mono.is_nondecreasing() and is_nonpositive(cos_arg):
                self.set_monotonicity(expr, Monotonicity.Nonincreasing)
            else:
                self.set_monotonicity(expr, Monotonicity.Unknown)

        elif name == 'cos':
            sin_arg = aml.sin(arg)
            if mono.is_nonincreasing() and is_nonnegative(sin_arg):
                self.set_monotonicity(expr, Monotonicity.Nondecreasing)
            elif mono.is_nondecreasing() and is_nonpositive(sin_arg):
                self.set_monotonicity(expr, Monotonicity.Nondecreasing)
            elif mono.is_nondecreasing() and is_nonnegative(sin_arg):
                self.set_monotonicity(expr, Monotonicity.Nonincreasing)
            elif mono.is_nonincreasing() and is_nonpositive(sin_arg):
                self.set_monotonicity(expr, Monotonicity.Nonincreasing)
            else:
                self.set_monotonicity(expr, Monotonicity.Unknown)

        else:
            raise RuntimeError('unknown unary function {}'.format(name))


def expr_monotonicity(expr):
    visitor = MonotonicityExprVisitor()
    visitor.visit(expr)
    return visitor.monotonicity(expr)


def is_nondecreasing(expr):
    return expr_monotonicity(expr).is_nondecreasing()


def is_nonincreasing(expr):
    return expr_monotonicity(expr).is_nonincreasing()


def is_unknown(expr):
    return expr_monotonicity(expr).is_unknown()

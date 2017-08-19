from numbers import Number
import pyomo.core.base.expr_pyomo4 as omo
from pyomo.core.base.var import SimpleVar


InequalityExpression = omo._InequalityExpression
EqualityExpression = omo._EqualityExpression
ProductExpression = omo._ProductExpression
DivisionExpression = omo._DivisionExpression
SumExpression = omo._SumExpression
LinearExpression = omo._LinearExpression
NegationExpression = omo._NegationExpression
UnaryFunctionExpression = omo._UnaryFunctionExpression
AbsExpression = omo._AbsExpression


def expr_callback(expr_cls):
    def decorator(func):
        func._expr_callback = expr_cls
        return func
    return decorator


class ExprVisitorMeta(type):
    def __new__(metacls, name, bases, namespace, **kwargs):
        result = type.__new__(metacls, name, bases, dict(namespace))
        result._expr_callbacks = [
            (f, f._expr_callback) for f in namespace.values() if hasattr(f, '_expr_callback')
        ]
        return result


def try_callback(visitor, expr):
    """Try calling one of the registered callbacks, raising an exception
    if no callback matches.
    """
    visited = False
    for cb, expr_cls in visitor._expr_callbacks:
        if isinstance(expr, expr_cls):
            cb(visitor, expr)
            visited = True
            break  # stop at first match

    if not visited:
        msg = 'No callback found for {} {}'.format(
            expr, type(expr)
        )
        raise RuntimeError(msg)


class ExprVisitor(object, metaclass=ExprVisitorMeta):
    def visit(self, expr):
        with omo.bypass_clone_check():
            stack = [(0, expr)]
            while len(stack) > 0:
                (lvl, expr) = stack.pop()

                if isinstance(expr, omo._ExpressionBase):
                    for arg in expr._args:
                        stack.append((lvl+1, arg))

                try_callback(self, expr)

            self.visit_end()

    def visit_end(self):
        pass


class BottomUpExprVisitor(object, metaclass=ExprVisitorMeta):
    """A visitor that visits leaf nodes first and the root last"""
    def visit(self, expr):
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
                try_callback(self, expr)

            self.visit_end()

    def visit_end(self):
        pass

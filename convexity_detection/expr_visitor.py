from numbers import Number
import pyomo.core.base.expr_pyomo4 as omo
from pyomo.core.base.var import SimpleVar


class ExprVisitor(object):
    def visit(self, expr):
        with omo.bypass_clone_check():
            stack = [(0, expr)]
            while len(stack) > 0:
                (lvl, expr) = stack.pop()
                if isinstance(expr, SimpleVar):
                    self.visit_simple_var(lvl, expr)
                    continue

                if isinstance(expr, omo._ExpressionBase):
                    for arg in expr._args:
                        stack.append((lvl+1, arg))

                if isinstance(expr, omo._InequalityExpression):
                    self.visit_inequality(expr)
                elif isinstance(expr, omo._EqualityExpression):
                    self.visit_equality(expr)
                elif isinstance(expr, omo._ProductExpression):
                    self.visit_product(lvl, expr)
                elif isinstance(expr, omo._DivisionExpression):
                    self.visit_division(lvl, expr)
                elif isinstance(expr, omo._SumExpression):
                    self.visit_sum(lvl, expr)
                elif isinstance(expr, omo._GetItemExpression):
                    self.visit_get_item(expr)
                elif isinstance(expr, omo.Expr_if):
                    self.visit_expr_if(expr)
                elif isinstance(expr, omo._ProductExpression):
                    self.visit_product(lvl, expr)
                elif isinstance(expr, omo._LinearExpression):
                    self.visit_linear(lvl, expr)
                elif isinstance(expr, omo._NegationExpression):
                    self.visit_negation(lvl, expr)
                elif isinstance(expr, omo._AbsExpression):
                    self.visit_abs(expr)
                elif isinstance(expr, omo._PowExpression):
                    self.visit_pow(expr)
                elif isinstance(expr, omo._UnaryFunctionExpression):
                    self.visit_unary_function(lvl, expr)
                elif isinstance(expr, omo._ExternalFunctionExpression):
                    self.visit_external_function(expr)
                elif isinstance(expr, Number):
                    self.visit_number(lvl, expr)
            self.visit_end()

    def visit_end(self):
        pass

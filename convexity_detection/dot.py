from numbers import Number
from convexity_detection.expr_visitor import (
    ExprVisitor,
    ProductExpression,
    DivisionExpression,
    SumExpression,
    LinearExpression,
    NegationExpression,
    UnaryFunctionExpression,
    expr_callback,
)
# import pyomo.core.base.expr_pyomo4 as omo
from pyomo.core.base.var import SimpleVar
import numpy as np


class DotVisitor(ExprVisitor):
    def __init__(self, outfile):
        self._f = outfile
        self._visited = set()
        self.write_header()

    def write_header(self):
        self._f.write('digraph G {\n')

    def write_node(self, node, label):
        if id(node) in self._visited:
            return
        self._visited.add(id(node))
        self._f.write('    {} [label="{}"];\n'.format(id(node), label))

    def write_coef_node(self, node, coef):
        if np.isclose(coef, 1.0):
            return node
        node_id = 'c{}'.format(node)
        node_prod_id = 'p{}'.format(node)
        self._f.write('    {} [label="{:.4f}"];\n'.format(node_id, coef))
        self._f.write('    {} [label="*"];\n'.format(node_prod_id))
        self._f.write('    {} -> {};\n'.format(node_prod_id, node))
        self._f.write('    {} -> {};\n'.format(node_prod_id, node_id))
        return node_prod_id

    def write_arc(self, from_, to):
        if isinstance(to, Number):
            self._f.write('    constant{0} [label="{0}"];\n'.format(to))
            self._f.write('    {} -> constant{};\n'.format(id(from_), to))
        else:
            self._f.write('    {} -> {};\n'.format(id(from_), id(to)))

    def visit_end(self):
        self._f.write('}\n')


    @expr_callback(ProductExpression)
    def visit_product(self, e):
        self.write_node(e, '*')
        for a in e._args:
            self.write_arc(e, a)

    @expr_callback(DivisionExpression)
    def visit_division(self, e):
        self.write_node(e, '/')
        for a in e._args:
            self.write_arc(e, a)

    @expr_callback(LinearExpression)
    def visit_linear(self, e):
        self.write_node(e, '+')

        for node, coef in e._coef.items():
            node_id = self.write_coef_node(node, coef)
            self._f.write('    {} -> {};\n'.format(id(e), node_id))

        if not np.isclose(e._const, 0):
            self._f.write('    const{} [label="{:.4f}"];\n'.format(id(e), e._const))
            self._f.write('    {0} -> const{0};\n'.format(id(e)))

    @expr_callback(SumExpression)
    def visit_sum(self, e):
        self.write_node(e, '+')
        for a in e._args:
            self.write_arc(e, a)

    @expr_callback(SimpleVar)
    def visit_simple_var(self, v):
        self.write_node(v, v.name)

    @expr_callback(NegationExpression)
    def visit_negation(self, e):
        self.write_node(e, 'neg')
        for a in e._args:
            self.write_arc(e, a)

    @expr_callback(UnaryFunctionExpression)
    def visit_unary_function(self, e):
        self.write_node(e, e._name)
        for a in e._args:
            self.write_arc(e, a)

    @expr_callback(Number)
    def visit_number(self, n):
        print(n)

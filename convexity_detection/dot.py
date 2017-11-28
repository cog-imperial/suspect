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
import convexity_detection.expr_visitor as expr_visitor
from pyomo.core.base.var import SimpleVar
import numpy as np


class DotExpressionHandler(expr_visitor.ExpressionHandler):
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

    def finish(self):
        self._f.write('}\n')

    def visit_equality(self, e):
        self.write_node(e, '=')
        for a in e._args:
            self.write_arc(e, a)
        return 42

    def visit_inequality(self, e):
        self.write_node(e, '<=>')
        for a in e._args:
            self.write_arc(e, a)

    def visit_product(self, e):
        self.write_node(e, '*')
        for a in e._args:
            self.write_arc(e, a)

    def visit_division(self, e):
        self.write_node(e, '/')
        for a in e._args:
            self.write_arc(e, a)

    def visit_linear(self, e):
        self.write_node(e, '$\sum$')

        for node, coef in e._coef.items():
            node_id = self.write_coef_node(node, coef)
            self._f.write('    {} -> {};\n'.format(id(e), node_id))

        if not np.isclose(e._const, 0):
            self._f.write('    const{} [label="{:.4f}"];\n'.format(
                id(e), e._const)
            )
            self._f.write('    {0} -> const{0};\n'.format(id(e)))

    def visit_sum(self, e):
        self.write_node(e, '+')
        for a in e._args:
            self.write_arc(e, a)

    def visit_variable(self, v):
        self.write_node(v, v.name)

    def visit_negation(self, e):
        self.write_node(e, 'neg')
        for a in e._args:
            self.write_arc(e, a)

    def visit_abs(self, e):
        self.write_node(e, 'abs')
        for a in e._args:
            self.write_arc(e, a)

    def visit_pow(self, e):
        self.write_node(e, '**')
        for a in e._args:
            self.write_arc(e, a)

    def visit_unary_function(self, e):
        self.write_node(e, e._name)
        for a in e._args:
            self.write_arc(e, a)

    def visit_number(self, n):
        print(n)

    def visit_numeric_constant(self, c):
        self._f.write('    {} [label="{:.4f}"];\n'.format(
            id(c), c.value))


def write_dot(outfile, expr):
    handler = DotExpressionHandler(outfile)
    expr_visitor.visit(handler, expr)
    handler.finish()

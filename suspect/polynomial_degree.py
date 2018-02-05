# Copyright 2018 Francesco Ceccon
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

import suspect.dag.expressions as dex
from suspect.dag.visitor import Dispatcher


def polynomial_degree(dag):
    visitor = PolynomialDegreeVisitor()
    dag.forward_visit(visitor)
    return visitor.result()


class PolynomialDegreeVisitor(object):
    def __init__(self):
        self._degree = {}
        self._dispatcher = Dispatcher(
            lookup={
                dex.Variable: self.visit_variable,
                dex.Constant: self.visit_constant,
                dex.Constraint: self.visit_constraint,
                dex.Objective: self.visit_objective,
                dex.ProductExpression: self.visit_product,
                dex.DivisionExpression: self.visit_division,
                dex.LinearExpression: self.visit_linear,
                dex.PowExpression: self.visit_pow,
                dex.SumExpression: self.visit_sum,
                dex.UnaryFunctionExpression: self.visit_unary_function,
            },
            allow_missing=False)

    def get(self, expr):
        return self._degree[id(expr)]

    def set(self, expr, value):
        self._degree[id(expr)] = value

    def result(self):
        return self._degree

    def visit_variable(self, _variable):
        return 1

    def visit_constant(self, _constant):
        return 0

    def visit_constraint(self, expr):
        return self.get(expr.children[0])

    def visit_objective(self, expr):
        return self.get(expr.children[0])

    def visit_product(self, expr):
        s = 0
        for child in expr.children:
            d = self.get(child)
            if d is None:
                return None
            s += d
        return s

    def visit_division(self, expr):
        assert len(expr.children) == 2
        den_degree = self.get(expr.children[1])
        if den_degree == 0:
            return self.get(expr.children[0])
        return None

    def visit_linear(self, expr):
        if len(expr.children) == 0:
            return 0
        return 1

    def visit_pow(self, expr):
        assert len(expr.children) == 2
        base, expo = expr.children
        base_degree = self.get(base)
        expo_degree = self.get(expo)
        if expo_degree == 0:
            if base_degree == 0:
                # const ** const
                return 0
            if not isinstance(expo, dex.Constant):
                return None
            expo = expo.value
            if expo == int(expo):
                if base_degree is not None and expo > 0:
                    return base_degree * expo
                elif expo == 0:
                    return 0
        return None

    def visit_sum(self, expr):
        # return max degree, or None
        ans = 0
        for child in expr.children:
            d = self.get(child)
            if d is None:
                return None
            if d > ans:
                ans = d
        return ans

    def visit_unary_function(self, _expr):
        return None

    def __call__(self, expr):
        new_value = self._dispatcher.dispatch(expr)
        self.set(expr, new_value)

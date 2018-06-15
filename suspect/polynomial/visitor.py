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

"""Visitor applying rules for polynomial degree computation."""

from suspect.polynomial.rules import *


class PolynomialDegreeVisitor(object):
    def __init__(self):
        self._rules = self.register_rules()
        self._callbacks = self._init_callbacks()

    def _init_callbacks(self):
        callbacks = {}
        for rule in self._rules:
            callbacks[rule.root_expr] = rule.apply
        return callbacks

    def handle_result(self, expr, result, ctx):
        """Handle visit result."""
        ctx[expr] = result
        return True

    def _handle_result(self, expr, result, ctx):
        return self.handle_result(expr, result, ctx)

    def _visit_expression(self, expr, ctx, callback):
        result = callback(expr, ctx)
        return self._handle_result(expr, result, ctx)

    def register_rules(self):
        return [
            VariableRule(),
            ConstantRule(),
            ConstraintRule(),
            ObjectiveRule(),
            DivisionRule(),
            ProductRule(),
            LinearRule(),
            SumRule(),
            NegationRule(),
            PowerRule(),
            UnaryFunctionRule(),
        ]

    def visit(self, expr, ctx):
        callback = self._callbacks.get(expr.expression_type)
        if callback is not None:
            return self._visit_expression(expr, ctx, callback)
        raise RuntimeError('visiting expression with no callback associated.')

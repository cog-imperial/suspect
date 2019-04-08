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

"""Base class for visitors applying the given rules."""
import abc
from suspect.pyomo.expressions import nonpyomo_leaf_types
from suspect.expression import ExpressionType


class Visitor(metaclass=abc.ABCMeta):
    """Base class for visitors applying rules to DAG vertices."""
    needs_matching_rules = True

    def __init__(self):
        self._rules = self.register_rules()
        self._callbacks = self._init_callbacks()

    @abc.abstractmethod
    def register_rules(self):
        """Register rules applied by this visitor."""
        pass

    @abc.abstractmethod
    def handle_result(self, expr, result, ctx):
        """Handle ``result`` obtained applying rule to ``expr``."""
        pass

    def visit(self, expr, ctx, *args):
        """Apply registered rule to ``expr``."""
        if type(expr) in nonpyomo_leaf_types:
            callback = self._callbacks.get(ExpressionType.Constant)
        elif expr.is_constant():
            callback = self._callbacks.get(ExpressionType.Constant)
        elif expr.is_variable_type():
            callback = self._callbacks.get(ExpressionType.Variable)
        else:
            callback = self._callbacks.get(expr.expression_type)

        if callback is not None:
            return self._visit_expression(expr, callback, ctx, *args)

        if self.needs_matching_rules:
            raise RuntimeError('visiting expression with no rule associated.')
        return False

    def _init_callbacks(self):
        callbacks = {}
        for expr_type, rule in self._rules.items():
            callbacks[expr_type] = rule.apply
        return callbacks

    def _handle_result(self, expr, result, ctx):
        return self.handle_result(expr, result, ctx)

    def _visit_expression(self, expr, callback, ctx, *args):
        result = callback(expr, ctx, *args)
        return self._handle_result(expr, result, ctx)


class ForwardVisitor(Visitor): # pylint: disable=abstract-method
    """Visitor when visiting DAG forward."""
    pass


class BackwardVisitor(Visitor): # pylint: disable=abstract-method
    """Visitor when visiting DAG backward."""
    def _handle_result(self, expr, result, ctx):
        if result is None:
            # if we allow matching rules, then continue iteration
            return not self.needs_matching_rules
        any_change = False
        if len(result) != len(expr.args):
            raise ValueError('Result must be a list with same length as expr.args')
        for child, value in zip(expr.args, result):
            any_change |= self.handle_result(child, value, ctx)
        return any_change

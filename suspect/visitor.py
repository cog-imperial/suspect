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


class Visitor(metaclass=abc.ABCMeta):
    """Base class for visitors applying rules to DAG vertices."""
    needs_matching_rules = True

    @abc.abstractmethod
    def visit_expression(self, expr, ctx, *args):
        """Apply rule to ``expr``.

        Parameters
        ----------
        expr : expression
            the expression being visited
        ctx : dict-like
            the current ctx
        *args : dict-like list
            additional ctx

        Returns
        -------
        bool
            Whether a rule was matched
        value
            The value that will be passed to ``handle_result``"""
        pass

    @abc.abstractmethod
    def handle_result(self, expr, result, ctx):
        """Handle ``result`` obtained applying rule to ``expr``."""
        pass

    def visit(self, expr, ctx, *args):
        """Apply registered rule to ``expr``."""
        return self._visit_expression(expr, ctx, *args)

    def _handle_result(self, expr, result, ctx):
        return self.handle_result(expr, result, ctx)

    def _visit_expression(self, expr, ctx, *args):
        matched, result = self.visit_expression(expr, ctx, *args)
        if not matched:
            if self.needs_matching_rules:
                raise RuntimeError('visiting expression with no rule associated.')
            return False
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
        if isinstance(result, list):
            if len(result) != len(expr.args):
                raise ValueError(
                    'Result must be a list with same length as expr.args'
                )
            for child, value in zip(expr.args, result):
                any_change |= self.handle_result(child, value, ctx)
            return any_change
        # Visitor returned a dict
        for arg, value in result.items():
            any_change |= self.handle_result(arg, value, ctx)
        return any_change

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

"""Interfaces used in SUSPECT.

You don't need them, but they are nice for documentation purpose.
"""
import abc
from suspect.expression import ExpressionType


class Problem(object): # pragma: no cover
    """Generic problem with vertices of type V."""
    pass


class Visitor(metaclass=abc.ABCMeta): # pragma: no cover
    """Visitor for vertices of Problem."""
    @abc.abstractmethod
    def visit(self, vertex, ctx):
        """Visit vertex. Return True if the vertex should be considered "dirty"."""
        pass


class Iterator(metaclass=abc.ABCMeta): # pragma: no cover
    """Iterator over vertices of Problem."""
    @abc.abstractmethod
    def iterate(self, problem, visitor, ctx, *args, **kwargs):
        """Iterate over vertices of problem, calling visitor on each one of them.

        Returns the list of vertices for which the visitor returned a True value.
        """
        pass


class ForwardIterator(Iterator): # pragma: no cover pylint: disable=abstract-method
    """An iterator for iterating over nodes in ascending depth order."""
    pass


class BackwardIterator(Iterator): # pragma: no cover pylint: disable=abstract-method
    """An iterator for iterating over nodes in descending depth order."""
    pass


class Rule(object): # pragma: no cover
    """Represent a series of contraints on an expression yielding a value."""
    root_expr = None

    def checked_apply(self, expr, ctx):
        """Apply rule to ``expr`` and ``ctx`` after checking matching ``expression_type``."""
        if expr.expression_type != self.root_expr:
            raise ValueError('expected {} expression type, but had {}'.format(
                self.root_expr, expr.expression_type
            ))
        return self.apply(expr, ctx)

    def apply(self, expr, ctx):
        """Apply rule to ``expr`` and ``ctx``."""
        pass


class UnaryFunctionRule(Rule): # pragma: no cover
    """Represent a series of contraints on an unary function expression yielding a value."""
    root_expr = ExpressionType.UnaryFunction
    func_type = None

    def checked_apply(self, expr, ctx):
        """Apply rule to ``expr`` and ``ctx`` after checking matching ``expression_type``."""
        if expr.expression_type != self.root_expr:
            raise ValueError('expected {} expression type, but had {}'.format(
                self.root_expr, expr.expression_type
            ))
        if expr.func_type != self.func_type:
            raise ValueError('expected {} function type, but had {}'.format(
                self.func_type, expr.func_type
            ))
        return self.apply(expr, ctx)


class CombineUnaryFunctionRules(Rule): # pragma: no cover
    """Rule to combine a collection of UnaryFunctionRule.

    Parameters
    ----------
    *args: UnaryFunctionRule list
       list of unary functions rules
    needs_matching_rules : bool
       if True, will raise an exception if no rule matched
    """
    root_expr = ExpressionType.UnaryFunction

    def __init__(self, *args, needs_matching_rules=True):
        self._rules = list(args)
        self._needs_matching_rules = needs_matching_rules
        self._apply_funcs = {}
        for rule in self._rules:
            if not rule.root_expr == ExpressionType.UnaryFunction:
                raise ValueError('Non unary function rule in CombineUnaryFunctionRules')
            self._apply_funcs[rule.func_type] = rule.apply

    def apply(self, expr, ctx):
        apply_func = self._apply_funcs.get(expr.func_type)
        if not apply_func:
            if not self._needs_matching_rules:
                return None
            raise RuntimeError(
                'Could not find rule for expression of expression_type={} and func_type={}'.format(
                    getattr(expr, 'expression_type'),
                    getattr(expr, 'func_type'),
                )
            )
        return apply_func(expr, ctx)

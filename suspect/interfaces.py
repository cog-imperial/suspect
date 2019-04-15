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
    def apply(self, expr, ctx):
        """Apply rule to ``expr`` and ``ctx``."""
        pass


class UnaryFunctionRule(Rule): # pragma: no cover
    """Represent a series of contraints on an unary function expression yielding a value."""
    pass


class CombineUnaryFunctionRules(Rule): # pragma: no cover
    """Rule to combine a collection of UnaryFunctionRule.

    Parameters
    ----------
    rules: UnaryFunctionRule dict
       dict of unary functions rules
    needs_matching_rules : bool
       if True, will raise an exception if no rule matched
    """
    def __init__(self, rules, needs_matching_rules=True):
        self._rules = rules
        self._needs_matching_rules = needs_matching_rules
        self._apply_funcs = {}
        for func_name, rule in self._rules.items():
            self._apply_funcs[func_name] = rule.apply

    def apply(self, expr, ctx, *args):
        apply_func = self._apply_funcs.get(expr.getname())
        if not apply_func:
            if not self._needs_matching_rules:
                return None
            if hasattr(expr, getname):
                func_name = expr.getname()
            else:
                func_name = 'Unknown'
            raise RuntimeError(
                'Could not find rule for expression of expression_type={} and func_name={}'.format(
                    getattr(expr, 'expression_type'),
                    func_name,
                )
            )
        return apply_func(expr, ctx, *args)

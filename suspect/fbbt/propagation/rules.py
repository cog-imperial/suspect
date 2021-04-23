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

"""FBBT bounds propagation rules."""
import operator
from functools import reduce
from suspect.pyomo.expressions import nonpyomo_leaf_types
from suspect.interval import Interval
from suspect.interfaces import Rule
from suspect.expression import ExpressionType, UnaryFunctionType
from suspect.math import almosteq # pylint: disable=no-name-in-module


class VariableRule(Rule):
    """Bound propagation rule for variables."""
    def apply(self, expr, _bounds):
        return Interval(expr.lb, expr.ub)


class ConstantRule(Rule):
    """Bound propagation rule for constants."""
    def apply(self, expr, _bounds):
        if type(expr) in nonpyomo_leaf_types:
            return Interval(expr, expr)
        return Interval(expr.value, expr.value)


class ConstraintRule(Rule):
    """Bound propagation rule for constraints."""
    def apply(self, expr, bounds):
        child = expr.args[0]
        child_bounds = bounds[child]
        constraint_bounds = Interval(expr.lower_bound, expr.upper_bound)
        return constraint_bounds.intersect(child_bounds)


class ObjectiveRule(Rule):
    """Bound propagation rule for objectives."""
    def apply(self, expr, bounds):
        child = expr.args[0]
        return bounds[child]


class ProductRule(Rule):
    """Bound propagation rule for products."""
    def apply(self, expr, bounds):
        children_bounds = [bounds[child] for child in expr.args]
        return reduce(operator.mul, children_bounds, 1.0)


class QuadraticRule(Rule):
    """Bound propagation rule for quadratic."""
    def apply(self, expr, bounds):
        return sum([self._term_bounds(term, bounds) for term in expr.terms])

    def _term_bounds(self, term, bounds):
        if id(term.var1) != id(term.var2):
            return bounds[term.var1] * bounds[term.var2] * term.coefficient
        return term.coefficient * (bounds[term.var1] ** 2)


class DivisionRule(Rule):
    """Bound propagation rule for divisions."""
    def apply(self, expr, bounds):
        assert len(expr.args) == 2
        num, den = expr.args
        return bounds[num] / bounds[den]


class ReciprocalRule(Rule):
    """Bound propagation rule for divisions."""
    def apply(self, expr, bounds):
        assert len(expr.args) == 1
        child = expr.args[0]
        return 1.0 / bounds[child]


class LinearRule(Rule):
    """Bound propagation rule for linear expressions."""
    def apply(self, expr, bounds):
        children_contribution = sum(
            expr.coefficient(child) * bounds[child]
            for child in expr.args
        )
        constant_contribution = Interval(expr.constant_term, expr.constant_term)
        return children_contribution + constant_contribution


class SumRule(Rule):
    """Bound propagation rule for sum."""
    def apply(self, expr, bounds):
        return sum(bounds[child] for child in expr.args)


class PowerRule(Rule):
    """Bound propagation rule for power."""
    def apply(self, expr, bounds):
        assert len(expr.args) == 2
        _, expo = expr.args
        if type(expo) not in nonpyomo_leaf_types:
            if not expo.is_constant():
                return Interval(None, None)
            expo = expo.value

        is_even = almosteq(expo % 2, 0)
        is_positive = expo > 0
        if is_even and is_positive:
            return Interval(0, None)
        return Interval(None, None)


class NegationRule(Rule):
    """Bound propagation rule for negation."""
    def apply(self, expr, bounds):
        return -bounds[expr.args[0]]


class ExpressionRule(Rule):
    """Bound propagation rule for Expressions"""
    def apply(self, expr, bounds):
        return bounds[expr.expr]


class _UnaryFunctionRule(Rule):
    """Bound propagation rule for unary functions."""
    def apply(self, expr, bounds):
        assert len(expr.args) == 1
        child = expr.args[0]
        child_bounds = bounds[child]
        func = getattr(child_bounds, self.func_name)
        return func()


class AbsRule(_UnaryFunctionRule):
    """Return new bounds for abs."""
    func_name = 'abs'


class SqrtRule(_UnaryFunctionRule):
    """Return new bounds for sqrt."""
    func_name = 'sqrt'


class ExpRule(_UnaryFunctionRule):
    """Return new bounds for exp."""
    func_name = 'exp'


class LogRule(_UnaryFunctionRule):
    """Return new bounds for log."""
    func_name = 'log'


class Log10Rule(_UnaryFunctionRule):
    """Return new bounds for log10."""
    func_name = 'log10'


class SinRule(_UnaryFunctionRule):
    """Return new bounds for sin."""
    func_name = 'sin'


class CosRule(_UnaryFunctionRule):
    """Return new bounds for cos."""
    func_name = 'cos'


class TanRule(_UnaryFunctionRule):
    """Return new bounds for tan."""
    func_name = 'tan'


class AsinRule(_UnaryFunctionRule):
    """Return new bounds for asin."""
    func_name = 'asin'


class AcosRule(_UnaryFunctionRule):
    """Return new bounds for acos."""
    func_name = 'acos'


class AtanRule(_UnaryFunctionRule):
    """Return new bounds for atan."""
    func_name = 'atan'


_func_name_to_rule_map = dict()
_func_name_to_rule_map['abs'] = AbsRule()
_func_name_to_rule_map['sqrt'] = SqrtRule()
_func_name_to_rule_map['exp'] = ExpRule()
_func_name_to_rule_map['log'] = LogRule()
_func_name_to_rule_map['log10'] = Log10Rule()
_func_name_to_rule_map['sin'] = SinRule()
_func_name_to_rule_map['cos'] = CosRule()
_func_name_to_rule_map['tan'] = TanRule()
_func_name_to_rule_map['asin'] = AsinRule()
_func_name_to_rule_map['acos'] = AcosRule()
_func_name_to_rule_map['atan'] = AtanRule()


class UnaryFunctionRule(Rule):
    """Bound propagation rule for unary functions."""
    def apply(self, expr, bounds):
        assert len(expr.args) == 1
        func_name = expr.getname()
        if func_name not in _func_name_to_rule_map:
            raise ValueError('Unknown function type {}'.format(func_name))
        return _func_name_to_rule_map[func_name].apply(expr, bounds)

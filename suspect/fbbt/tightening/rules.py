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

"""FBBT bounds tightening rules."""
import numpy as np
from suspect.pyomo.expressions import nonpyomo_leaf_types
from suspect.interfaces import Rule, UnaryFunctionRule
from suspect.expression import ExpressionType, UnaryFunctionType
from suspect.interval import Interval
from suspect.math import almosteq, inf # pylint: disable=no-name-in-module


MAX_EXPR_CHILDREN = 1000


class ConstraintRule(Rule):
    """Return new bounds for constraint."""
    def apply(self, expr, _bounds):
        child = expr.args[0]
        bounds = Interval(expr.lower_bound, expr.upper_bound)
        return [bounds]


class SumRule(Rule):
    """Return new bounds for sum."""
    def apply(self, expr, bounds):
        if len(expr.args) > MAX_EXPR_CHILDREN: # pragma: no cover
            return None
        expr_bound = bounds[expr]
        children_bounds = [
            self._child_bounds(child, siblings, expr_bound, bounds)
            for child, siblings in _sum_child_and_siblings(expr.args)
        ]
        return children_bounds

    def _child_bounds(self, child, siblings, expr_bound, bounds):
        siblings_bound = sum(bounds[s] for s in siblings)
        return expr_bound - siblings_bound


class LinearRule(Rule):
    """Return new bounds for linear expressions."""
    def apply(self, expr, bounds):
        if len(expr.args) > MAX_EXPR_CHILDREN: # pragma: no cover
            return None
        expr_bound = bounds[expr]
        const = expr.constant_term
        coefficients = [expr.coefficient(ch) for ch in expr.args]
        children_bounds = [
            self._child_bounds(child, child_c, siblings, const, expr_bound, bounds)
            for (child_c, child), siblings
            in _linear_child_and_siblings(coefficients, expr.args)
        ]
        return children_bounds

    def _child_bounds(self, child, coef, siblings, const, expr_bound, bounds):
        siblings_bound = sum(bounds[s] * c for c, s in siblings) + const
        return (expr_bound - siblings_bound) / coef



class QuadraticRule(Rule):
    """Return new bounds for quadratic expressions."""
    def apply(self, expr, ctx):
        raise NotImplementedError('QuadraticRule.apply')
        expr_bound = ctx.bounds(expr)
        child_bounds = {}
        for term, siblings in self._quadratic_term_and_siblings(expr):
            var1 = term.var1
            var2 = term.var2
            siblings_bound = sum(self._term_bound(t, ctx) for t in siblings)
            term_bound = (expr_bound - siblings_bound) / term.coefficient
            if var1 == var2:
                term_bound = term_bound.intersect(Interval(0, None))
                upper_bound = term_bound.sqrt().upper_bound
                new_bound = Interval(-upper_bound, upper_bound)

                if var1 in child_bounds:
                    existing = child_bounds[var1]
                    child_bounds[var1] = existing.intersect(new_bound)
                else:
                    child_bounds[var1] = Interval(new_bound.lower_bound, new_bound.upper_bound)

            else:
                new_bound_var1 = term_bound / ctx.bounds(var2)
                new_bound_var2 = term_bound / ctx.bounds(var1)

                if var1 in child_bounds:
                    existing = child_bounds[var1]
                    child_bounds[var1] = existing.intersect(new_bound_var1)
                else:
                    child_bounds[var1] = new_bound_var1

                if var2 in child_bounds:
                    existing = child_bounds[var2]
                    child_bounds[var2] = existing.intersect(new_bound_var2)
                else:
                    child_bounds[var2] = new_bound_var2

        return child_bounds

    def _quadratic_term_and_siblings(self, expr):
        terms = expr.terms
        for i, term in enumerate(terms):
            yield term, terms[:i] + terms[i+1:]

    def _term_bound(self, term, ctx):
        if term.var1 == term.var2:
            return term.coefficient * (ctx.bounds(term.var1) ** 2)
        return term.coefficient * ctx.bounds(term.var1) * ctx.bounds(term.var2)


class PowerRule(Rule):
    """Return new bounds for power expressions."""
    def apply(self, expr, bounds):
        base, expo = expr.args
        if type(expo) not in nonpyomo_leaf_types:
            if not expo.is_constant():
                return None
            expo = expo.value

        if not almosteq(expo, 2):
            return None

        expr_bound = bounds[expr]
        # the bound of a square number is never negative, but check anyway to
        # avoid unexpected crashes.
        if not expr_bound.is_nonnegative():
            return None

        sqrt_bound = expr_bound.sqrt()
        return [
            Interval(-sqrt_bound.upper_bound, sqrt_bound.upper_bound),
            None,
        ]


class _UnaryFunctionRule(UnaryFunctionRule):
    def apply(self, expr, bounds):
        expr_bounds = bounds[expr]
        return [self._child_bounds(expr_bounds)]

    def _child_bounds(self, bounds):
        pass


class _BoundedFunctionRule(_UnaryFunctionRule):
    func_name = None

    def _child_bounds(self, bounds):
        func = getattr(bounds, self.func_name)
        return func.inverse()


class AbsRule(_BoundedFunctionRule):
    """Return new bounds for abs."""
    func_name = 'abs'


class SqrtRule(_BoundedFunctionRule):
    """Return new bounds for sqrt."""
    func_name = 'sqrt'


class ExpRule(_BoundedFunctionRule):
    """Return new bounds for exp."""
    func_name = 'exp'


class LogRule(_BoundedFunctionRule):
    """Return new bounds for log."""
    func_name = 'log'


class _UnboundedFunctionRule(_UnaryFunctionRule):
    def _child_bounds(self, bounds):
        return Interval(None, None)


class SinRule(_UnboundedFunctionRule):
    """Return new bounds for sin."""
    func_type = UnaryFunctionType.Sin


class CosRule(_UnboundedFunctionRule):
    """Return new bounds for cos."""
    func_type = UnaryFunctionType.Cos


class TanRule(_UnboundedFunctionRule):
    """Return new bounds for tan."""
    func_type = UnaryFunctionType.Tan


class AsinRule(_UnboundedFunctionRule):
    """Return new bounds for asin."""
    func_type = UnaryFunctionType.Asin


class AcosRule(_UnboundedFunctionRule):
    """Return new bounds for acos."""
    func_type = UnaryFunctionType.Acos


class AtanRule(_UnboundedFunctionRule):
    """Return new bounds for atan."""
    func_type = UnaryFunctionType.Atan


_func_name_to_rule_map = dict()
_func_name_to_rule_map['abs'] = AbsRule()
_func_name_to_rule_map['sqrt'] = SqrtRule()
_func_name_to_rule_map['exp'] = ExpRule()
_func_name_to_rule_map['log'] = LogRule()
_func_name_to_rule_map['sin'] = SinRule()
_func_name_to_rule_map['cos'] = CosRule()
_func_name_to_rule_map['tan'] = TanRule()
_func_name_to_rule_map['asin'] = AsinRule()
_func_name_to_rule_map['acos'] = AcosRule()
_func_name_to_rule_map['atan'] = AtanRule()


class UnaryFunctionRule(Rule):
    """Bound tightening rule for unary functions."""
    def apply(self, expr, bounds):
        assert len(expr.args) == 1
        func_name = expr.getname()
        if func_name not in _func_name_to_rule_map:
            raise ValueError('Unknown function type {}'.format(func_name))
        return _func_name_to_rule_map[func_name].apply(expr, bounds)


def _sum_child_and_siblings(children):
    for i, _ in enumerate(children):
        yield children[i], children[:i] + children[i+1:]


def _linear_child_and_siblings(coefficients, children):
    for i, child in enumerate(children):
        child_c = coefficients[i]
        other_children = np.concatenate((children[:i], children[i+1:]))
        other_coefficients = np.concatenate((coefficients[:i], coefficients[i+1:]))
        yield (child_c, child), zip(other_coefficients, other_children)

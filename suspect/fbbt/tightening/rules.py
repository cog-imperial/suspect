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
from pyomo.common.collections import ComponentMap
import pyomo.environ as pe
from suspect.expression import UnaryFunctionType
from suspect.interfaces import Rule, UnaryFunctionRule
from suspect.interval import Interval
from suspect.math import almosteq, isinf  # pylint: disable=no-name-in-module
from suspect.pyomo.expressions import nonpyomo_leaf_types

MAX_EXPR_CHILDREN = 1000


class ConstraintRule(Rule):
    """Return new bounds for constraint."""
    def apply(self, expr, _bounds):
        bounds = Interval(expr.lower_bound, expr.upper_bound)
        return [bounds]


class SumRule(Rule):
    """Return new bounds for sum."""
    def __init__(self, max_expr_children=None):
        if max_expr_children is None:
            max_expr_children = MAX_EXPR_CHILDREN
        self.max_expr_children = max_expr_children

    def apply(self, expr, bounds):
        if len(expr.args) > self.max_expr_children:  # pragma: no cover
            return None
        expr_bound = bounds[expr]
        children_bounds = [
            self._child_bounds(child_idx, child, expr, expr_bound, bounds)
            for child_idx, child in enumerate(expr.args)
        ]
        return children_bounds

    def _child_bounds(self, child_idx, _child, expr, expr_bound, bounds):
        siblings_bound = sum(
            bounds[c]
            for i, c in enumerate(expr.args)
            if i != child_idx
        )
        return expr_bound - siblings_bound


class LinearRule(Rule):
    """Return new bounds for linear expressions."""
    def __init__(self, max_expr_children=None):
        if max_expr_children is None:
            max_expr_children = MAX_EXPR_CHILDREN
        self.max_expr_children = max_expr_children

    def apply(self, expr, bounds):
        if expr.nargs() > self.max_expr_children:  # pragma: no cover
            return None
        expr_bound = bounds[expr]
        children_bounds = [
            bounds[child] * coef
            for coef, child in zip(expr.linear_coefs, expr.linear_vars)
        ]

        const = expr.constant
        children_bounds = [
            self._child_bounds(child_idx, coef, const, expr_bound, children_bounds)
            for child_idx, (coef, child) in enumerate(zip(expr.linear_coefs, expr.linear_vars))
        ]
        return children_bounds

    def _child_bounds(self, child_idx, coef, const, expr_bound, children_bounds):
        siblings_bound = sum(
            b for i, b in enumerate(children_bounds) if i != child_idx
        ) + const
        return (expr_bound - siblings_bound) / coef


class QuadraticRule(Rule):
    """Return new bounds for quadratic expressions."""
    def __init__(self, max_expr_children=None):
        if max_expr_children is None:
            max_expr_children = MAX_EXPR_CHILDREN
        self.max_expr_children = max_expr_children

    def apply(self, expr, bounds):
        expr_bound = bounds[expr]
        child_bounds = ComponentMap()
        if len(expr.terms) > self.max_expr_children:
            return None

        # Build bounds for all terms
        terms_bounds = [self._term_bound(t, bounds) for t in expr.terms]

        terms = expr.terms

        for term_idx, term in enumerate(terms):
            var1 = term.var1
            var2 = term.var2
            siblings_bound = sum(
                terms_bounds[i]
                for i, t in enumerate(terms) if i != term_idx
            )
            term_bound = (expr_bound - siblings_bound) / term.coefficient
            if id(var1) == id(var2):
                term_bound = term_bound.intersect(Interval(0, None))
                upper_bound = term_bound.sqrt().upper_bound
                new_bound = Interval(-upper_bound, upper_bound)

                if var1 in child_bounds:
                    existing = child_bounds[var1]
                    child_bounds[var1] = existing.intersect(new_bound)
                else:
                    child_bounds[var1] = Interval(new_bound.lower_bound, new_bound.upper_bound)

            else:
                new_bound_var1 = term_bound / bounds[var2]
                new_bound_var2 = term_bound / bounds[var1]

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

    def _term_bound(self, term, bounds):
        if term.var1 is term.var2:
            return term.coefficient * (bounds[term.var1] ** 2)
        return term.coefficient * bounds[term.var1] * bounds[term.var2]


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


class MonomialTermRule(Rule):
    """Return new bounds for monomial term expressions."""
    def apply(self, expr, bounds):
        expr_bound = bounds[expr]
        const, expr = expr.args
        const = pe.value(const)
        if almosteq(const, 0.0):
            return None
        child_bounds = ComponentMap()
        child_bounds[expr] = expr_bound / const
        return child_bounds


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

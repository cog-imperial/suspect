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
from typing import List, MutableMapping
from pyomo.core.expr.numeric_expr import NumericValue
from suspect.pyomo.quadratic import BilinearTerm


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

    def apply(self, expr, bounds: MutableMapping[NumericValue, Interval]):
        bnds_list: List[Interval] = list()
        bnds_list.append(bounds[expr.arg(0)])
        expr_bounds: Interval = bounds[expr]
        for i in range(1, expr.nargs()):
            last_bounds = bnds_list[i-1]
            arg_bounds = bounds[expr.arg(i)]
            bnds_list.append(last_bounds + arg_bounds)
        if expr_bounds.lower_bound > bnds_list[-1].lower_bound:
            bnds_list[-1] = Interval(expr_bounds.lower_bound, bnds_list[-1].upper_bound)
        if expr_bounds.upper_bound < bnds_list[-1].upper_bound:
            bnds_list[-1] = Interval(bnds_list[-1].lower_bound, expr_bounds.upper_bound)

        children_bounds = [None] * expr.nargs()
        for i in reversed(range(1, expr.nargs())):
            bnds0 = bnds_list[i]
            bnds1 = bnds_list[i-1]
            bnds2 = bounds[expr.arg(i)]
            _bnds1 = bnds0 - bnds2
            _bnds2 = bnds0 - bnds1
            bnds1 = bnds1.intersect(_bnds1)
            bnds2 = bnds2.intersect(_bnds2)
            bnds_list[i-1] = bnds1
            children_bounds[i] = bnds2

        bnds = bounds[expr.arg(0)]
        _bnds = bnds_list[0]
        bnds = bnds.intersect(_bnds)
        children_bounds[0] = bnds

        return children_bounds


class LinearRule(Rule):
    """Return new bounds for linear expressions."""
    def __init__(self, max_expr_children=None):
        if max_expr_children is None:
            max_expr_children = MAX_EXPR_CHILDREN
        self.max_expr_children = max_expr_children

    def apply(self, expr, bounds):
        bnds_list: List[Interval] = list()
        const_val = pe.value(expr.constant)
        bnds_list.append(Interval(const_val, const_val))
        expr_bounds: Interval = bounds[expr]
        for coef, v in zip(expr.linear_coefs, expr.linear_vars):
            last_bounds = bnds_list[-1]
            coef = pe.value(coef)
            v_bounds = bounds[v]
            bnds_list.append(last_bounds + coef*v_bounds)
        if expr_bounds.lower_bound > bnds_list[-1].lower_bound:
            bnds_list[-1] = Interval(expr_bounds.lower_bound, bnds_list[-1].upper_bound)
        if expr_bounds.upper_bound < bnds_list[-1].upper_bound:
            bnds_list[-1] = Interval(bnds_list[-1].lower_bound, expr_bounds.upper_bound)

        children_bounds = [None] * len(expr.linear_vars)
        for i in reversed(range(len(expr.linear_vars))):
            bnds0 = bnds_list[i + 1]
            bnds1 = bnds_list[i]
            coef = pe.value(expr.linear_coefs[i])
            v = expr.linear_vars[i]
            v_bounds = bounds[v]
            bnds2 = coef * v_bounds
            _bnds1 = bnds0 - bnds2
            _bnds2 = bnds0 - bnds1
            bnds1 = bnds1.intersect(_bnds1)
            bnds2 = bnds2.intersect(_bnds2)
            bnds_list[i] = bnds1
            children_bounds[i] = bnds2 / coef

        return children_bounds


def _update_var_bounds_from_bilinear_term_bounds(t: BilinearTerm,
                                                 term_bound: Interval,
                                                 bounds: MutableMapping[NumericValue, Interval],
                                                 child_bounds: MutableMapping[NumericValue, Interval]):
    term_bound = term_bound / t.coefficient
    if t.var1 is t.var2:
        term_bound = term_bound.intersect(Interval(0, None))
        upper_bound = term_bound.sqrt().upper_bound
        new_bound = Interval(-upper_bound, upper_bound)
        if t.var1 in child_bounds:
            existing = child_bounds[t.var1]
            child_bounds[t.var1] = existing.intersect(new_bound)
        else:
            child_bounds[t.var1] = new_bound
    else:
        new_bound_var1 = term_bound / bounds[t.var2]
        new_bound_var2 = term_bound / bounds[t.var1]

        if t.var1 in child_bounds:
            existing = child_bounds[t.var1]
            child_bounds[t.var1] = existing.intersect(new_bound_var1)
        else:
            child_bounds[t.var1] = new_bound_var1

        if t.var2 in child_bounds:
            existing = child_bounds[t.var2]
            child_bounds[t.var2] = existing.intersect(new_bound_var2)
        else:
            child_bounds[t.var2] = new_bound_var2


class QuadraticRule(Rule):
    """Return new bounds for quadratic expressions."""
    def __init__(self, max_expr_children=None):
        if max_expr_children is None:
            max_expr_children = MAX_EXPR_CHILDREN
        self.max_expr_children = max_expr_children

    def apply(self, expr, bounds):
        bnds_list: List[Interval] = list()
        expr_bounds: Interval = bounds[expr]
        terms_bounds = [self._term_bound(t, bounds) for t in expr.terms]
        bnds_list.append(terms_bounds[0])
        for t_bnds in terms_bounds[1:]:
            bnds_list.append(bnds_list[-1] + t_bnds)
        if expr_bounds.lower_bound > bnds_list[-1].lower_bound:
            bnds_list[-1] = Interval(expr_bounds.lower_bound, bnds_list[-1].upper_bound)
        if expr_bounds.upper_bound < bnds_list[-1].upper_bound:
            bnds_list[-1] = Interval(bnds_list[-1].lower_bound, expr_bounds.upper_bound)

        terms = list(expr.terms)
        child_bounds = ComponentMap()
        for i in reversed(range(1, len(terms_bounds))):
            bnds0 = bnds_list[i]
            bnds1 = bnds_list[i-1]
            bnds2 = terms_bounds[i]
            _bnds1 = bnds0 - bnds2
            _bnds2 = bnds0 - bnds1
            bnds1 = bnds1.intersect(_bnds1)
            bnds2 = bnds2.intersect(_bnds2)
            bnds_list[i-1] = bnds1
            _update_var_bounds_from_bilinear_term_bounds(t=terms[i],
                                                         term_bound=bnds2,
                                                         bounds=bounds,
                                                         child_bounds=child_bounds)

        bnds = terms_bounds[0]
        _bnds = bnds_list[0]
        bnds = bnds.intersect(_bnds)
        _update_var_bounds_from_bilinear_term_bounds(t=terms[0],
                                                     term_bound=bnds,
                                                     bounds=bounds,
                                                     child_bounds=child_bounds)

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

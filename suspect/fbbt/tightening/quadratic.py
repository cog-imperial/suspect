#  Copyright 2019 Francesco Ceccon
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Bounds tightening of univariate quadratic expression"""
from collections import namedtuple
from pyomo.common.collections import ComponentMap
from suspect.math import (
    almosteq, isinf, inf, sqrt, RoundMode as RM
)
from suspect.expression import ExpressionType
from suspect.interval import Interval
from suspect.interfaces import Rule


_VariableCoefficients = \
    namedtuple('_VariableCoefficients', ['quadratic', 'linear'])


class UnivariateQuadraticRule(Rule):
    """Bounds tightening of univariate quadratic expression.

    References
    ----------
    [1] Decomposition of multistage stochastic programs and a constraint integer
    programming approach to mixed-integer nonlinear programming. S. Vigerske.
    PhD thesis, Humboldt Universität zu Berlin
    """
    def __init__(self):
        pass

    def apply(self, expr, bounds):
        # Look for Quadratic + Linear expressions
        if expr.nargs() != 2:
            return

        quadratic, linear = _quadratic_and_linear(expr.children)

        if quadratic is None or linear is None:
            return

        univariate_terms, quadratic_terms, linear_terms = \
            _collect_expression_types(quadratic, linear)

        # Compute bounds of non univariate terms. We will use this to perform
        # tightening on univariate terms.
        quadratic_terms_bound = sum(
            t.coefficient * bounds[t.var1] * bounds[t.var2]
            for t in quadratic_terms
        )
        linear_terms_bound = sum(
            bounds[v] * c
            for v, c in linear_terms
        )
        expr_bound = bounds.get(expr, Interval(None, None))

        non_univariate_bound = \
            expr_bound - quadratic_terms_bound - linear_terms_bound

        univariate_terms_bounds = [
            _univariate_term_bound(a, b, bounds[v])
            for (v, a, b) in univariate_terms
        ]

        tightened_univariate_terms_bounds = [
            _tighten_univariate_term_bound(
                term_idx, univariate_terms_bounds, non_univariate_bound
            )
            for term_idx, _ in enumerate(univariate_terms_bounds)
        ]

        vars_bounds = ComponentMap(
            (v, _variable_bound_from_univariate_bound(a, b, bound))
            for bound, (v, a, b)
            in zip(tightened_univariate_terms_bounds, univariate_terms)
        )

        return vars_bounds


def _variable_bound_from_univariate_bound(a, b, bound):
    bounds_from_lower_bound = _variable_bound_from_univariate_lower_bound(
        a, b, bound.lower_bound
    )
    bounds_from_upper_bound = _variable_bound_from_univariate_lower_bound(
        -a, -b, -bound.upper_bound
    )
    return bounds_from_lower_bound.intersect(bounds_from_upper_bound)


def _variable_bound_from_univariate_lower_bound(a, b, c):
    t = c/a + (b**2) / (4*a**2)

    if a > 0:
        return Interval(None, None)
        # if c + (b**2) / (4*a) <= 0:
        #     return Interval(None, None)
        # new_upper_bound = Interval(None, -sqrt(t, RM.RN) - b/(2*a))
        # new_lower_bound = Interval(sqrt(t, RM.RN) - b/(2*a), None)
        # return new_upper_bound.intersect(new_lower_bound)

    # a < 0
    if c + (b**2) / (4*a) > 0:
        # The bound would be empty which is something that shouldn't happen.
        # Return [-inf, inf] so we don't tighten the bounds
        return Interval(None, None)

    return Interval(-sqrt(t, RM.RN) - b/(2*a), sqrt(t, RM.RN) - b/(2*a))


def _tighten_univariate_term_bound(term_idx, univariate_terms_bounds,
                                   expr_bounds):
    siblings_bound = sum(
        bound
        for i, bound in enumerate(univariate_terms_bounds)
        if i != term_idx
    )
    return expr_bounds - siblings_bound


def _univariate_term_bound(a, b, x_bound):
    """Return the bound of the univariate term ax^2 + bx"""
    lower_bound = x_bound.lower_bound
    upper_bound = x_bound.upper_bound

    if lower_bound is None or isinf(lower_bound):
        univariate_lower_bound = -inf
    else:
        univariate_lower_bound = a * lower_bound**2 + b*lower_bound

    if upper_bound is None or isinf(upper_bound):
        univariate_upper_bound = inf
    else:
        univariate_upper_bound = a * upper_bound**2 + b*upper_bound

    if -b/(2*a) in x_bound:
        t = -(b**2) / (4*a)
        new_lower_bound = min(univariate_lower_bound, univariate_upper_bound, t)
        new_upper_bound = max(univariate_lower_bound, univariate_upper_bound, t)
        return Interval(new_lower_bound, new_upper_bound)
    new_lower_bound = min(univariate_lower_bound, univariate_upper_bound)
    new_upper_bound = max(univariate_lower_bound, univariate_upper_bound)
    return Interval(new_lower_bound, new_upper_bound)


def _collect_expression_types(quadratic, linear):
    """Collect different expression types from quadratic and linear expression.

    Given an expression

    a0 x0^2 + a1 x1^2  + ...+ b0 x0 + b1 x1 + b2 x2 + ...

    Returns the quadratic univariate expressions like `a0 x0^2`, the bilinear
    expressions `c x1 x2`, and the linear expressions `bn xn`.

    :param quadratic: the quadratic expression
    :param linear: the linear expression
    :return:
    A tuple (univariate_expr, bilinear_terms, linear_terms)
    """
    variables_coef = ComponentMap()

    for coef, var in zip(linear.linear_coefs, linear.linear_vars):
        variables_coef[var] = \
            _VariableCoefficients(quadratic=0.0, linear=coef)

    non_univariate_terms = []
    for term in quadratic.terms:
        if term.var1 is term.var2:
            var_coef = variables_coef.get(term.var1, None)
            if var_coef is None:
                non_univariate_terms.append(term)
            else:
                linear_coef = var_coef.linear
                var_coef = _VariableCoefficients(
                    quadratic=term.coefficient, linear=linear_coef
                )
                variables_coef[term.var1] = var_coef
        else:
            non_univariate_terms.append(term)

    linear_terms = [
        (v, vc.linear)
        for v, vc in variables_coef.items()
        if almosteq(vc.quadratic, 0.0)
    ]
    univariate_terms = [
        (v, vc.quadratic, vc.linear)
        for v, vc in variables_coef.items()
        if not almosteq(vc.quadratic, 0.0)
    ]

    return univariate_terms, non_univariate_terms, linear_terms


def _quadratic_and_linear(children):
    assert len(children) == 2
    a, b = children
    if a.expression_type == ExpressionType.Quadratic:
        if b.expression_type == ExpressionType.Linear:
            return a, b
        return None, None

    if b.expression_type == ExpressionType.Quadratic:
        if a.expression_type == ExpressionType.Linear:
            return b, a

    return None, None

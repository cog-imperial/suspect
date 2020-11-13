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

"""Convexity detection rules for product expressions."""
from suspect.pyomo.expressions import MonomialTermExpression, nonpyomo_leaf_types
from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule
from suspect.expression import ExpressionType


class ProductRule(ConvexityRule):
    """Return convexity of product."""
    def apply(self, expr, convexity, monotonicity, bounds):
        if isinstance(expr, MonomialTermExpression):
            _, var = expr.args
            return convexity[var]

        f, g = expr.args
        if f is g:
            return _product_as_square_convexity(f, convexity, bounds)

        # -c*x*x is encoded as Linear([-c], [x])*Variable(x)
        g_is_var = type(g) not in nonpyomo_leaf_types and g.is_variable_type()
        f_is_var = type(f) not in nonpyomo_leaf_types and f.is_variable_type()

        if isinstance(f, MonomialTermExpression) and g_is_var:
            cvx = _product_monomial_term_by_variable_convexity(f, g)
            if cvx is not None:
                return cvx

        if f_is_var and isinstance(g, MonomialTermExpression):
            cvx = _product_monomial_term_by_variable_convexity(g, f)
            if cvx is not None:
                return cvx

        mono_g = monotonicity[g]
        mono_f = monotonicity[f]
        if mono_g.is_constant():
            return _product_convexity(f, g, convexity, bounds)
        elif mono_f.is_constant():
            return _product_convexity(g, f, convexity, bounds)
        return Convexity.Unknown


def _product_as_square_convexity(f, convexity, bounds):
    # same as f**2
    cvx_f = convexity[f]
    bounds_f = bounds.get(f)
    if cvx_f.is_linear():
        return Convexity.Convex
    elif bounds_f is None:
        return Convexity.Unknown
    elif cvx_f.is_convex() and bounds_f.is_nonnegative():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_f.is_nonpositive():
        return Convexity.Convex
    return Convexity.Unknown


def _product_monomial_term_by_variable_convexity(f, g):
    coef, f_v = f.args
    if f_v is g:
        if coef > 0:
            return Convexity.Convex
        return Convexity.Concave
    return None


def _product_convexity(f, g, convexity, bounds):
    cvx_f = convexity[f]
    bounds_g = bounds.get(g)

    if cvx_f.is_linear():
        return cvx_f
    elif bounds_g is None:
        return Convexity.Unknown
    elif cvx_f.is_convex() and bounds_g.is_nonnegative():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_g.is_nonpositive():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_g.is_nonnegative():
        return Convexity.Concave
    elif cvx_f.is_convex() and bounds_g.is_nonpositive():
        return Convexity.Concave
    return Convexity.Unknown

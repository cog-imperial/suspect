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
from suspect.convexity.convexity import Convexity
from suspect.expression import ExpressionType
from suspect.interfaces import Rule


class ProductRule(Rule):
    """Return convexity of product."""
    root_expr = ExpressionType.Product

    def apply(self, expr, ctx):
        f, g = expr.children
        if f is g:
            return _product_as_square_convexity(f, ctx)

        # -c*x*x is encoded as Linear([-c], [x])*Variable(x)
        if f.expression_type == ExpressionType.Linear and g.expression_type == ExpressionType.Variable:
            cvx = _product_linear_by_variable_convexity(f, g)
            if cvx is not None:
                return cvx

        if g.expression_type == ExpressionType.Linear and f.expression_type == ExpressionType.Variable:
            cvx = _product_linear_by_variable_convexity(g, f)
            if cvx is not None:
                return cvx

        mono_g = ctx.monotonicity(g)
        mono_f = ctx.monotonicity(f)
        if mono_g.is_constant():
            return _product_convexity(f, g, ctx)
        elif mono_f.is_constant():
            return _product_convexity(g, f, ctx)
        return Convexity.Unknown


def _product_as_square_convexity(f, ctx):
    # same as f**2
    cvx_f = ctx.convexity(f)
    bounds_f = ctx.bounds(f)
    if cvx_f.is_linear():
        return Convexity.Convex
    elif cvx_f.is_convex() and bounds_f.is_nonnegative():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_f.is_nonpositive():
        return Convexity.Convex
    return Convexity.Unknown


def _product_linear_by_variable_convexity(f, g):
    if len(f.children) == 1 and f.children[0] is g:
        if f.coefficients[0] > 0:
            return Convexity.Convex
        return Convexity.Concave
    return None


def _product_convexity(f, g, ctx):
    cvx_f = ctx.convexity(f)
    bounds_g = ctx.bounds(g)

    if cvx_f.is_linear():
        return cvx_f
    elif cvx_f.is_convex() and bounds_g.is_nonnegative():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_g.is_nonpositive():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_g.is_nonnegative():
        return Convexity.Concave
    elif cvx_f.is_convex() and bounds_g.is_nonpositive():
        return Convexity.Concave
    return Convexity.Unknown

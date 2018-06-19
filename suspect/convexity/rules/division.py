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

"""Convexity detection rules for division expressions."""
from suspect.convexity.convexity import Convexity
from suspect.expression import ExpressionType
from suspect.interfaces import Rule


class DivisionRule(Rule):
    """Return convexity of division."""
    root_expr = ExpressionType.Division

    def apply(self, expr, ctx):
        f, g = expr.children
        return _division_convexity(f, g, ctx)


def _division_convexity(f, g, ctx):
    mono_f = ctx.monotonicity(f)
    mono_g = ctx.monotonicity(g)

    bounds_f = ctx.bounds(f)
    bounds_g = ctx.bounds(g)

    if mono_g.is_constant():
        cvx_f = ctx.convexity(f)
        return _division_convexity_constant_g(mono_f, mono_g, bounds_f, bounds_g, cvx_f)
    elif mono_f.is_constant():
        cvx_g = ctx.convexity(g)
        return _division_convexity_constant_f(mono_f, mono_g, bounds_f, bounds_g, cvx_g)
    return Convexity.Unknown


def _division_convexity_constant_g(_mono_f, _mono_g, _bounds_f, bounds_g, cvx_f):
    if cvx_f.is_convex() and bounds_g.is_positive():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_g.is_negative():
        return Convexity.Convex
    elif cvx_f.is_concave() and bounds_g.is_positive():
        return Convexity.Concave
    elif cvx_f.is_convex() and bounds_g.is_negative():
        return Convexity.Concave
    return Convexity.Unknown


def _division_convexity_constant_f(_mono_f, _mono_g, bounds_f, bounds_g, cvx_g):
    # want to avoid g == 0
    if 0 in bounds_g:
        return Convexity.Unknown

    if bounds_f.is_nonnegative():
        if bounds_g.is_positive() and cvx_g.is_concave():
            return Convexity.Convex
        elif bounds_g.is_negative() and cvx_g.is_convex():
            return Convexity.Concave
    elif bounds_f.is_nonpositive():
        if bounds_g.is_negative() and cvx_g.is_convex():
            return Convexity.Convex
        elif bounds_g.is_positive() and cvx_g.is_concave():
            return Convexity.Concave
    return Convexity.Unknown

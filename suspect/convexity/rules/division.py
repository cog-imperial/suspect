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
from suspect.convexity.rules.rule import ConvexityRule
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.interval import Interval


class DivisionRule(ConvexityRule):
    """Return convexity of division."""
    def apply(self, expr, convexity, monotonicity, bounds):
        f, g = expr.args
        cvx_f = convexity[f]
        cvx_g = convexity[g]

        mono_f = monotonicity[f]
        mono_g = monotonicity[g]

        bounds_f = bounds.get(f)
        bounds_g = bounds.get(g)

        return _division_convexity(cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g)


class ReciprocalRule(ConvexityRule):
    """Return convexity of reciprocal."""
    def apply(self, expr, convexity, monotonicity, bounds):
        g = expr.args[0]
        cvx_f = Convexity.Linear
        cvx_g = convexity[g]

        mono_f = Monotonicity.Constant
        mono_g = monotonicity[g]

        bounds_f = Interval(1.0, 1.0)
        bounds_g = bounds.get(g)

        return _division_convexity(cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g)


def _division_convexity(cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g):
    if mono_g.is_constant():
        return _division_convexity_constant_g(mono_f, mono_g, bounds_f, bounds_g, cvx_f)
    elif mono_f.is_constant():
        return _division_convexity_constant_f(mono_f, mono_g, bounds_f, bounds_g, cvx_g)
    return Convexity.Unknown


def _division_convexity_constant_g(_mono_f, _mono_g, _bounds_f, bounds_g, cvx_f):
    if bounds_g is None:
        return Convexity.Unknown
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
    if bounds_g is None or bounds_f is None:
        return Convexity.Unknown

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

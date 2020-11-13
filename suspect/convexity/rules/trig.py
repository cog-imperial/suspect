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

"""Convexity detection rules for trigonometric expressions."""
from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule
from suspect.interval import Interval
from suspect.math import pi # pylint: disable=no-name-in-module


class SinRule(ConvexityRule):
    """Return convexity of sin."""
    def apply(self, expr, convexity, _mono, bounds): # pylint: disable=too-many-return-statements
        child = expr.args[0]
        child_bounds = bounds.get(child)
        cvx = convexity[child]

        if child_bounds is None:
            return Convexity.Unknown

        if child_bounds.size() > pi:
            return Convexity.Unknown

        sin_bounds = child_bounds.sin()
        if sin_bounds.lower_bound * sin_bounds.upper_bound < 0:
            return Convexity.Unknown

        cos_bounds = child_bounds.cos()

        if sin_bounds.is_nonnegative():
            if cvx.is_linear():
                return Convexity.Concave
            if cvx.is_convex() and cos_bounds.is_nonpositive():
                return Convexity.Concave
            if cvx.is_concave() and cos_bounds.is_nonnegative():
                return Convexity.Concave
        elif sin_bounds.is_nonpositive():
            if cvx.is_linear():
                return Convexity.Convex
            if cvx.is_concave() and cos_bounds.is_nonpositive():
                return Convexity.Convex
            if cvx.is_convex() and cos_bounds.is_nonnegative():
                return Convexity.Convex

        return Convexity.Unknown


class AsinRule(ConvexityRule):
    """Return convexity of asin."""
    def apply(self, expr, convexity, _mono, bounds):
        child = expr.args[0]
        child_bounds = bounds.get(child)
        cvx = convexity[child]

        if child_bounds is None:
            return Convexity.Unknown

        concave_domain = Interval(-1, 0)
        if child_bounds in concave_domain and cvx.is_concave():
            return Convexity.Concave

        convex_domain = Interval(0, 1)
        if child_bounds in convex_domain and cvx.is_convex():
            return Convexity.Convex

        return Convexity.Unknown



class CosRule(ConvexityRule):
    """Return convexity of cos."""
    def apply(self, expr, convexity, _mono, bounds): # pylint: disable=too-many-return-statements
        child = expr.args[0]
        child_bounds = bounds.get(child)
        cvx = convexity[child]

        if child_bounds is None:
            return Convexity.Unknown

        if child_bounds.size() > pi:
            return Convexity.Unknown

        cos_bounds = child_bounds.cos()
        if cos_bounds.lower_bound * cos_bounds.upper_bound < 0:
            return Convexity.Unknown

        sin_bounds = child_bounds.sin()

        if cos_bounds.is_nonnegative():
            if cvx.is_linear():
                return Convexity.Concave
            if cvx.is_convex() and sin_bounds.is_nonpositive():
                return Convexity.Concave
            if cvx.is_concave() and sin_bounds.is_nonnegative():
                return Convexity.Concave
        elif cos_bounds.is_nonpositive():
            if cvx.is_linear():
                return Convexity.Convex
            if cvx.is_concave() and sin_bounds.is_nonpositive():
                return Convexity.Convex
            if cvx.is_convex() and sin_bounds.is_nonnegative():
                return Convexity.Convex

        return Convexity.Unknown


class AcosRule(ConvexityRule):
    """Return convexity of acos."""
    def apply(self, expr, convexity, _mono, bounds):
        child = expr.args[0]
        child_bounds = bounds.get(child)
        cvx = convexity[child]

        if child_bounds is None:
            return Convexity.Unknown

        convex_domain = Interval(-1, 0)
        if child_bounds in convex_domain and cvx.is_concave():
            return Convexity.Convex

        concave_domain = Interval(0, 1)
        if child_bounds in concave_domain and cvx.is_convex():
            return Convexity.Concave

        return Convexity.Unknown


class TanRule(ConvexityRule):
    """Return convexity of tan."""
    def apply(self, expr, convexity, _mono, bounds):
        child = expr.args[0]
        child_bounds = bounds.get(child)
        cvx = convexity[child]

        if child_bounds is None:
            return Convexity.Unknown

        if 2.0*child_bounds.size() > pi:
            return Convexity.Unknown

        tan_bounds = child_bounds.tan()
        if tan_bounds.lower_bound * tan_bounds.upper_bound < 0:
            return Convexity.Unknown

        if tan_bounds.is_nonnegative() and cvx.is_convex():
            return Convexity.Convex

        if tan_bounds.is_nonpositive() and cvx.is_concave():
            return Convexity.Concave

        return Convexity.Unknown


class AtanRule(ConvexityRule):
    """Return convexity of atan."""
    def apply(self, expr, convexity, _mono, bounds):
        child = expr.args[0]
        child_bounds = bounds.get(child)
        cvx = convexity[child]

        if child_bounds is None:
            return Convexity.Unknown

        if child_bounds.is_nonpositive() and cvx.is_convex():
            return Convexity.Convex

        if child_bounds.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave

        return Convexity.Unknown

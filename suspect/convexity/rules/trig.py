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
from suspect.expression import UnaryFunctionType
from suspect.interfaces import UnaryFunctionRule
from suspect.interval import Interval
from suspect.math import pi # pylint: disable=no-name-in-module


class SinRule(UnaryFunctionRule):
    """Return convexity of sin."""
    func_type = UnaryFunctionType.Sin

    def apply(self, expr, ctx): # pylint: disable=too-many-return-statements
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)

        if bounds.size() > pi:
            return Convexity.Unknown

        sin_bounds = bounds.sin()
        if sin_bounds.lower_bound * sin_bounds.upper_bound < 0:
            return Convexity.Unknown

        cos_bounds = bounds.cos()

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


class AsinRule(UnaryFunctionRule):
    """Return convexity of asin."""
    func_type = UnaryFunctionType.Asin

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)

        concave_domain = Interval(-1, 0)
        if bounds in concave_domain and cvx.is_concave():
            return Convexity.Concave

        convex_domain = Interval(0, 1)
        if bounds in convex_domain and cvx.is_convex():
            return Convexity.Convex

        return Convexity.Unknown



class CosRule(UnaryFunctionRule):
    """Return convexity of cos."""
    func_type = UnaryFunctionType.Cos

    def apply(self, expr, ctx): # pylint: disable=too-many-return-statements
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)

        if bounds.size() > pi:
            return Convexity.Unknown

        cos_bounds = bounds.cos()
        if cos_bounds.lower_bound * cos_bounds.upper_bound < 0:
            return Convexity.Unknown

        sin_bounds = bounds.sin()

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


class AcosRule(UnaryFunctionRule):
    """Return convexity of acos."""
    func_type = UnaryFunctionType.Acos

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)

        convex_domain = Interval(-1, 0)
        if bounds in convex_domain and cvx.is_concave():
            return Convexity.Convex

        concave_domain = Interval(0, 1)
        if bounds in concave_domain and cvx.is_convex():
            return Convexity.Concave

        return Convexity.Unknown


class TanRule(UnaryFunctionRule):
    """Return convexity of tan."""
    func_type = UnaryFunctionType.Tan

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)

        if 2.0*bounds.size() > pi:
            return Convexity.Unknown

        tan_bounds = bounds.tan()
        if tan_bounds.lower_bound * tan_bounds.upper_bound < 0:
            return Convexity.Unknown

        if tan_bounds.is_nonnegative() and cvx.is_convex():
            return Convexity.Convex

        if tan_bounds.is_nonpositive() and cvx.is_concave():
            return Convexity.Concave

        return Convexity.Unknown


class AtanRule(UnaryFunctionRule):
    """Return convexity of atan."""
    func_type = UnaryFunctionType.Atan

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)

        if bounds.is_nonpositive() and cvx.is_convex():
            return Convexity.Convex

        if bounds.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave

        return Convexity.Unknown

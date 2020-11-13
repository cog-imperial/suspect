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

"""Convexity detection rules for power expressions."""
from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule
from suspect.pyomo.expressions import nonpyomo_leaf_types
from suspect.math import almosteq, almostgte # pylint: disable=no-name-in-module


class PowerRule(ConvexityRule):
    """Return monotonicity of power expression."""
    def apply(self, expr, convexity, monotonicity, bounds):
        assert len(expr.args) == 2
        base, expo = expr.args

        mono_base = monotonicity[base]
        mono_expo = monotonicity[expo]

        if mono_expo.is_constant():
            if type(expo) not in nonpyomo_leaf_types:
                expo = expo.value
            cvx_base = convexity[base]
            bounds_base = bounds.get(base)
            return _constant_expo_pow_convexity(expo, cvx_base, bounds_base)
        elif mono_base.is_constant():
            if type(base) not in nonpyomo_leaf_types:
                base = base.value
            cvx_expo = convexity[expo]
            return _constant_base_pow_convexity(base, cvx_expo)
        return Convexity.Unknown


def _constant_base_pow_convexity(base, cvx_expo):
    if 0 < base < 1:
        if cvx_expo.is_concave():
            return Convexity.Convex
    elif almostgte(base, 1):
        if cvx_expo.is_convex():
            return Convexity.Convex
    return Convexity.Unknown


def _constant_expo_pow_convexity(expo, cvx_base, bounds_base):
    is_integer = almosteq(expo, int(expo))
    is_even = almosteq(expo % 2, 0)

    if almosteq(expo, 0):
        return Convexity.Linear
    elif almosteq(expo, 1):
        return cvx_base
    elif is_integer and is_even:
        return _integer_even_expo_pow_convexity(expo, cvx_base, bounds_base)
    elif is_integer:
        return _integer_odd_expo_pow_convexity(expo, cvx_base, bounds_base)
    return _noninteger_expo_pow_convexity(expo, cvx_base, bounds_base)


def _integer_even_expo_pow_convexity(expo, cvx_base, bounds_base):
    if expo > 0:
        return _integer_positive_even_expo_pow_convexity(expo, cvx_base, bounds_base)
    return _integer_negative_even_expo_pow_convexity(expo, cvx_base, bounds_base)


def _integer_positive_even_expo_pow_convexity(_expo, cvx_base, bounds_base):
    if cvx_base.is_linear():
        return Convexity.Convex
    elif bounds_base is None:
        return Convexity.Unknown
    elif cvx_base.is_convex() and bounds_base.is_nonnegative():
        return Convexity.Convex
    elif cvx_base.is_concave() and bounds_base.is_nonpositive():
        return Convexity.Convex
    return Convexity.Unknown


def _integer_negative_even_expo_pow_convexity(_expo, cvx_base, bounds_base):
    if bounds_base is None:
        return Convexity.Unknown
    elif cvx_base.is_convex() and bounds_base.is_nonpositive():
        return Convexity.Convex
    elif cvx_base.is_concave() and bounds_base.is_nonnegative():
        return Convexity.Convex
    elif cvx_base.is_convex() and bounds_base.is_nonnegative():
        return Convexity.Concave
    elif cvx_base.is_concave() and bounds_base.is_nonpositive():
        return Convexity.Concave
    return Convexity.Unknown


def _integer_odd_expo_pow_convexity(expo, cvx_base, bounds_base):
    if expo > 0:
        return _integer_positive_odd_expo_pow_convexity(expo, cvx_base, bounds_base)
    return _integer_negative_odd_expo_pow_convexity(expo, cvx_base, bounds_base)


def _integer_positive_odd_expo_pow_convexity(expo, cvx_base, bounds_base):
    if almosteq(expo, 1):  # pragma: no cover
        return cvx_base  # we won't reach this because of the check in apply.
    elif bounds_base is None:
        return Convexity.Unknown
    elif cvx_base.is_convex() and bounds_base.is_nonnegative():
        return Convexity.Convex
    elif cvx_base.is_concave() and bounds_base.is_nonpositive():
        return Convexity.Concave
    return Convexity.Unknown


def _integer_negative_odd_expo_pow_convexity(_expo, cvx_base, bounds_base):
    if bounds_base is None:
        return Convexity.Unknown
    elif cvx_base.is_concave() and bounds_base.is_nonnegative():
        return Convexity.Convex
    elif cvx_base.is_convex() and bounds_base.is_nonpositive():
        return Convexity.Concave
    return Convexity.Unknown


def _noninteger_expo_pow_convexity(expo, cvx_base, bounds_base):
    if bounds_base is None:
        return Convexity.Unknown
    if not bounds_base.is_nonnegative():
        return Convexity.Unknown
    if cvx_base.is_convex() and expo > 1:
        return Convexity.Convex
    elif cvx_base.is_concave() and expo < 0:
        return Convexity.Convex
    elif cvx_base.is_concave() and 0 < expo < 1:
        return Convexity.Concave
    elif cvx_base.is_convex() and expo < 0:
        return Convexity.Concave
    return Convexity.Unknown

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

"""Monotonicity detection rules for power expressions."""
from suspect.math import almosteq, \
    almostgte  # pylint: disable=no-name-in-module
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.rules.rule import MonotonicityRule
from suspect.pyomo.expressions import nonpyomo_leaf_types


class PowerRule(MonotonicityRule):
    """Return monotonicity of power expression."""
    def apply(self, expr, monotonicity, bounds):
        base, expo = expr.args

        mono_base = monotonicity[base]
        mono_expo = monotonicity[expo]

        bounds_base = bounds.get(base)
        bounds_expo = bounds.get(expo)

        if mono_base.is_constant():
            if type(base) not in nonpyomo_leaf_types:
                base = base.value
            return _monotonicity_constant_base(
                base, expo,
                mono_base, mono_expo,
                bounds_base, bounds_expo,
            )
        elif mono_expo.is_constant():
            if type(expo) not in nonpyomo_leaf_types:
                expo = expo.value
            return _monotonicity_constant_exponent(
                base, expo,
                mono_base, mono_expo,
                bounds_base, bounds_expo,
            )
        return Monotonicity.Unknown


def _monotonicity_constant_base(base, _expo, _mono_base, mono_expo, _bounds_base, bounds_expo):
    # pylint: disable=too-many-return-statements, too-many-arguments
    if base < 0:
        return Monotonicity.Unknown
    elif almosteq(base, 0):
        return Monotonicity.Constant
    elif 0 < base < 1:
        if bounds_expo is None:
            return Monotonicity.Unknown
        if mono_expo.is_nondecreasing() and bounds_expo.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono_expo.is_nonincreasing() and bounds_expo.is_nonnegative():
            return Monotonicity.Nondecreasing
        return Monotonicity.Unknown
    elif almostgte(base, 1):
        if bounds_expo is None:
            return Monotonicity.Unknown
        if mono_expo.is_nondecreasing() and bounds_expo.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono_expo.is_nonincreasing() and bounds_expo.is_nonpositive():
            return Monotonicity.Nondecreasing
        return Monotonicity.Unknown
    return Monotonicity.Unknown  # pragma: no cover


def _monotonicity_constant_exponent(base, expo, mono_base, mono_expo, bounds_base, bounds_expo):
    # pylint: disable=too-many-return-statements, too-many-arguments
    if almosteq(expo, 1):
        return mono_base
    elif almosteq(expo, 0):
        return Monotonicity.Constant
    is_integer = almosteq(expo, int(expo))
    is_even = almosteq(expo % 2, 0)
    if is_integer and is_even:
        return _monotonicity_even_exponent(
            base, expo, mono_base, mono_expo, bounds_base, bounds_expo,
        )
    elif is_integer:  # is odd
        return _monotonicity_odd_exponent(
            base, expo, mono_base, mono_expo, bounds_base, bounds_expo,
        )
    return _monotonicity_noninteger_exponent(
        base, expo, mono_base, mono_expo, bounds_base, bounds_expo,
    )


def _monotonicity_even_exponent(base, expo, mono_base, mono_expo, bounds_base, bounds_expo):
    # pylint: disable=too-many-return-statements, too-many-arguments, no-else-return
    if expo > 0:
        return _monotonicity_even_positive_exponent(
            base, expo, mono_base, mono_expo, bounds_base, bounds_expo,
        )
    else:
        return _monotonicity_even_negative_exponent(
            base, expo, mono_base, mono_expo, bounds_base, bounds_expo,
        )


def _monotonicity_even_positive_exponent(_base, _expo, mono_base, _mono_expo,
                                         bounds_base, _bounds_expo):
    if bounds_base is None:
        return Monotonicity.Unknown
    # pylint: disable=too-many-return-statements, too-many-arguments
    if mono_base.is_nondecreasing() and bounds_base.is_nonnegative():
        return Monotonicity.Nondecreasing
    elif mono_base.is_nonincreasing() and bounds_base.is_nonpositive():
        return Monotonicity.Nondecreasing
    elif mono_base.is_nondecreasing() and bounds_base.is_nonpositive():
        return Monotonicity.Nonincreasing
    elif mono_base.is_nonincreasing() and bounds_base.is_nonnegative():
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown  # pragma: no cover


def _monotonicity_even_negative_exponent(_base, _expo, mono_base, _mono_expo,
                                         bounds_base, _bounds_expo):
    if bounds_base is None:
        return Monotonicity.Unknown
    # pylint: disable=too-many-return-statements, too-many-arguments
    if mono_base.is_nonincreasing() and bounds_base.is_nonnegative():
        return Monotonicity.Nondecreasing
    elif mono_base.is_nondecreasing() and bounds_base.is_nonpositive():
        return Monotonicity.Nondecreasing
    elif mono_base.is_nonincreasing() and bounds_base.is_nonpositive():
        return Monotonicity.Nonincreasing
    elif mono_base.is_nondecreasing() and bounds_base.is_nonnegative():
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown  # pragma: no cover


def _monotonicity_odd_exponent(_base, expo, mono_base, _mono_expo, _bounds_base, _bounds_expo):
    # pylint: disable=too-many-return-statements, too-many-arguments
    if expo > 0 and mono_base.is_nondecreasing():
        return Monotonicity.Nondecreasing
    elif expo < 0 and mono_base.is_nonincreasing():
        return Monotonicity.Nondecreasing
    elif expo > 0 and mono_base.is_nonincreasing():
        return Monotonicity.Nonincreasing
    elif expo < 0 and mono_base.is_nondecreasing():
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown # pragma: no cover


def _monotonicity_noninteger_exponent(_base, expo, mono_base, _mono_expo,
                                      bounds_base, _bounds_expo):
    # pylint: disable=too-many-return-statements, too-many-arguments
    if bounds_base is None or not bounds_base.is_nonnegative():
        return Monotonicity.Unknown
    elif expo > 0:
        return mono_base
    elif expo < 0:
        if mono_base.is_nondecreasing():
            return Monotonicity.Nonincreasing
        elif mono_base.is_nonincreasing():
            return Monotonicity.Nondecreasing
    return Monotonicity.Unknown  # pragma: no cover

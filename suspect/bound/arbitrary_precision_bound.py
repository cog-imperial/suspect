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

from numbers import Number
from suspect.error import DomainError
from suspect.math.arbitrary_precision import (
    inf,
    mpf,
    isnan,
    almostlte,
    almostgte,
    almosteq,
)
from suspect.bound.bound import Bound


class ArbitraryPrecisionBound(Bound):
    def __init__(self, lower, upper):
        if lower is None:
            lower = -inf
        if upper is None:
            upper = inf

        lower = mpf(lower)
        upper = mpf(upper)

        if lower > upper:
            raise ValueError('lower must be <= upper')

        self.lower = lower
        self.upper = upper

    @property
    def lower_bound(self):
        return self.lower

    @property
    def upper_bound(self):
        return self.upper

    def is_zero(self):
        return almosteq(self.lower, 0) and almosteq(self.upper, 0)

    def is_positive(self):
        return self.lower > 0

    def is_negative(self):
        return self.upper < 0

    def is_nonpositive(self):
        return almostlte(self.upper, 0)

    def is_nonnegative(self):
        return almostgte(self.lower, 0)


    def tighten(self, other):
        if not isinstance(other, ArbitraryPrecisionBound):
            raise ValueError('other must be ArbitraryPrecisionBound')

        if other.lower < self.lower:
            new_l = self.lower
        else:
            new_l = other.lower

        if other.upper > self.upper:
            new_u = self.upper
        else:
            new_u = other.upper

        if new_u < new_l:
            raise DomainError(
                "Invalid tightened bound. This probably means the "
                "resulting domain is empty. This is not currently "
                "supported as it should not happen."
            )

        return ArbitraryPrecisionBound(new_l, new_u)

    def add(self, other):
        l = self.lower
        u = self.upper
        if isinstance(other, ArbitraryPrecisionBound):
            return ArbitraryPrecisionBound(l + other.lower, u + other.upper)
        elif isinstance(other, Number):
            return ArbitraryPrecisionBound(l + other, u + other)
        else:
            raise TypeError(
                "adding ArbitraryPrecisionBound to incompatbile type"
            )

    def sub(self, other):
        l = self.lower
        u = self.upper
        if isinstance(other, ArbitraryPrecisionBound):
            return ArbitraryPrecisionBound(l - other.upper, u - other.lower)
        elif isinstance(other, Number):
            return ArbitraryPrecisionBound(l - other, u - other)
        else:
            raise TypeError(
                "subtracting ArbitraryPrecisionBound to incompatbile type"
            )

    def mul(self, other):
        l = self.lower
        u = self.upper
        if isinstance(other, ArbitraryPrecisionBound):
            # Check for zero to handle case we or other has an
            # infinite bound.
            if self.is_zero() or other.is_zero():
                return self.zero()
            ol = other.lower
            ou = other.upper
            # 0*inf returns nan. We want to avoid it so that we can
            # return reasonable values
            candidates = [
                c for c in [l*ol, l*ou, u*ol, u*ou]
                if not isnan(c)
            ]
            new_l = min(candidates)
            new_u = max(candidates)
            return ArbitraryPrecisionBound(new_l, new_u)
        elif isinstance(other, Number):
            return self.__mul__(ArbitraryPrecisionBound(other, other))
        else:
            raise TypeError(
                "multiplying ArbitraryPrecisionBound to incompatible type"
            )

    def div(self, other):
        if isinstance(other, ArbitraryPrecisionBound):
            ol = other.lower
            ou = other.upper
            if almostlte(ol, 0) and almostgte(ou, 0):
                return ArbitraryPrecisionBound(-inf, inf)
            else:
                return self.__mul__(ArbitraryPrecisionBound(1/ou, 1/ol))
        elif isinstance(other, Number):
            return self.__truediv__(ArbitraryPrecisionBound(other, other))
        else:
            raise TypeError(
                "dividing AribtraryPrecisionBound by incompatible type"
            )

        pass

    def equals(self, other):
        if not isinstance(other, ArbitraryPrecisionBound):
            return False
        return (
            almosteq(self.lower, other.lower) and
            almosteq(self.upper, other.upper)
        )

    def contains(self, other):
        if isinstance(other, Number):
            return (
                almostgte(other, self.lower) and
                almostlte(other, self.upper)
            )
        elif isinstance(other, ArbitraryPrecisionBound):
            return (
                almostgte(other.lower, self.lower) and
                almostlte(other.upper, self.upper)
            )
        else:
            raise TypeError(
                "comparing ArbitraryPrecisionBound by incompatible type"
            )

    def zero(self):
        return ArbitraryPrecisionBound(0, 0)

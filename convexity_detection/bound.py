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
from convexity_detection.math import (
    inf,
    mpf,
    almostlte,
    almostgte,
    almosteq,
)


class Bound(object):
    def __init__(self, l, u):
        if l is None:
            l = -inf
        if u is None:
            u = inf

        if not isinstance(l, mpf):
            l = mpf(l)
        if not isinstance(u, mpf):
            u = mpf(u)

        if l > u:
            raise ValueError('l must be >= u')

        self.l = l
        self.u = u

    def is_zero(self):
        return almosteq(self.l, 0) and almosteq(self.u, 0)

    def is_positive(self):
        return self.l > 0

    def is_negative(self):
        return self.u < 0

    def is_nonnegative(self):
        return almostgte(self.l, 0)

    def is_nonpositive(self):
        return almostlte(self.u, 0)

    def tighten(self, other):
        """Returns a new bound which is the tightest intersection of bounds."""
        if other.l < self.l:
            new_l = self.l
        else:
            new_l = other.l

        if other.u > self.u:
            new_u = self.u
        else:
            new_u = other.u

        return Bound(new_l, new_u)

    def __add__(self, other):
        l = self.l
        u = self.u
        if isinstance(other, Bound):
            return Bound(l + other.l, u + other.u)
        elif isinstance(other, Number):
            return Bound(l + other, u + other)
        else:
            raise TypeError('adding Bound to incompatbile type')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        l = self.l
        u = self.u
        if isinstance(other, Bound):
            return Bound(l - other.u, u - other.l)
        elif isinstance(other, Number):
            return Bound(l - other, u - other)
        else:
            raise TypeError('subtracting Bound to incompatbile type')

    def __mul__(self, other):
        l = self.l
        u = self.u
        if isinstance(other, Bound):
            ol = other.l
            ou = other.u
            new_l = min(l * ol, l * ou, u * ol, u * ou)
            new_u = max(l * ol, l * ou, u * ol, u * ou)
            return Bound(new_l, new_u)
        elif isinstance(other, Number):
            return self.__mul__(Bound(other, other))
        else:
            raise TypeError('multiplying Bound to incompatible type')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Bound):
            ol = other.l
            ou = other.u
            if ol <= 0 and ou >= 0:
                return Bound(-inf, inf)
            else:
                return self.__mul__(Bound(1/ou, 1/ol))
        elif isinstance(other, Number):
            return self.__truediv__(Bound(other, other))
        else:
            raise TypeError('dividing Bound by incompatible type')

    def __eq__(self, other):
        if not isinstance(other, Bound):
            return False
        return almosteq(self.l, other.l) and almosteq(self.u, other.u)

    def __contains__(self, other):
        return (
            almostgte(other.l, self.l) and
            almostlte(other.u, self.u)
        )

    def __repr__(self):
        return '<{} at {}>'.format(str(self), id(self))

    def __str__(self):
        return '[{}, {}]'.format(self.l, self.u)

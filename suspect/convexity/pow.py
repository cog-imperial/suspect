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

from suspect.util import numeric_types, numeric_value
from suspect.math import almosteq, almostlte, almostgte
from suspect.convexity.convexity import Convexity


def _numeric_exponent_pow_convexity(handler, base, exponent):
    is_integer = almosteq(exponent, int(exponent))
    is_even = almosteq(exponent % 2, 0)

    cvx = handler.convexity(base)

    if is_integer and is_even and exponent > 0:
        if cvx.is_linear():
            return Convexity.Convex
        elif cvx.is_convex() and handler.is_nonnegative(base):
            return Convexity.Convex
        elif cvx.is_concave() and handler.is_nonpositive(base):
            return Convexity.Convex

    elif is_integer and is_even and almostlte(exponent, 0):
        if cvx.is_convex() and handler.is_nonpositive(base):
            return Convexity.Convex
        elif cvx.is_concave() and handler.is_nonnegative(base):
            return Convexity.Convex
        elif cvx.is_convex() and handler.is_nonnegative(base):
            return Convexity.Concave
        elif cvx.is_concave() and handler.is_nonpositive(base):
            return Convexity.Concave

    elif is_integer and exponent > 0:  # odd
        if almosteq(exponent, 1):
            return cvx
        elif cvx.is_convex() and handler.is_nonnegative(base):
            return Convexity.Convex
        elif cvx.is_concave() and handler.is_nonpositive(base):
            return Convexity.Concave

    elif is_integer and almostlte(exponent, 0):  # odd negative
        if cvx.is_concave() and handler.is_nonnegative(base):
            return Convexity.Convex
        elif cvx.is_convex() and handler.is_nonpositive(base):
            return Convexity.Concave

    else:
        # not integral
        if handler.is_nonnegative(base):
            if cvx.is_convex() and exponent > 1:
                return Convexity.Convex
            elif cvx.is_concave() and exponent < 0:
                return Convexity.Convex
            elif cvx.is_concave() and 0 < exponent < 1:
                return Convexity.Concave
            elif cvx.is_convex() and exponent < 0:
                return Convexity.Concave

    return Convexity.Unknown


def pow_convexity(handler, expr):
    assert len(expr._args) == 2
    base, exponent = expr._args

    if isinstance(exponent, numeric_types):
        return _numeric_exponent_pow_convexity(
            handler,
            base,
            numeric_value(exponent),
        )
    elif isinstance(base, numeric_types):
        base = numeric_value(base)
        cvx = handler.convexity(exponent)
        if 0 < base < 1:
            if cvx.is_concave():
                return Convexity.Convex

        elif almostgte(base, 1):
            if cvx.is_convex():
                return Convexity.Convex
            else:
                return Convexity.Unknown

    return Convexity.Unknown

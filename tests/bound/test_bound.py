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

import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from suspect.bound import ArbitraryPrecisionBound


@st.composite
def reals(draw, min_value=None, max_value=None):
    return draw(st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=False
    ))


@st.composite
def arbitrary_precision_bounds(draw, allow_infinity=True,
                               lower_bound=None, upper_bound=None):
    if lower_bound is None and allow_infinity:
        lower = draw(st.one_of(st.none(), reals(max_value=upper_bound)))
    else:
        lower = draw(reals(min_value=lower_bound, max_value=upper_bound))

    if upper_bound is None and allow_infinity:
        upper = draw(st.one_of(st.none(), reals(min_value=lower)))
    else:
        upper = draw(reals(min_value=lower, max_value=upper_bound))
    return ArbitraryPrecisionBound(lower, upper)


class TestAddition(object):
    @given(arbitrary_precision_bounds())
    def test_addition_with_zero(self, bound):
        zero_bound = ArbitraryPrecisionBound(0, 0)
        assert bound + zero_bound == bound

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_commutative_property(self, a, b):
        assert a + b == b + a

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(lower_bound=0),
    )
    def test_addition_with_positive_bound(self, a, b):
        c = a + b
        s = ArbitraryPrecisionBound(a.lower_bound, None)
        assert c in s

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(upper_bound=0),
    )
    def test_addition_with_negative_bound(self, a, b):
        c = a + b
        s = ArbitraryPrecisionBound(None, a.upper_bound)
        assert c in s

    @given(
        arbitrary_precision_bounds(),
        reals(),
    )
    def test_addition_with_floats(self, a, f):
        b = ArbitraryPrecisionBound(f, f)
        assert a + f == a + b


class TestSubtraction(object):
    @given(
        arbitrary_precision_bounds(),
    )
    def test_negation(self, a):
        assert (-(-a)) == a

    @given(
        arbitrary_precision_bounds(),
    )
    def test_subtraction_with_zero(self, a):
        assert a - 0 == a

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(lower_bound=0),
    )
    def test_subtraction_with_positive_bound(self, a, b):
        c = a - b
        s = ArbitraryPrecisionBound(None, a.upper_bound)
        assert c in s

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(upper_bound=0),
    )
    def test_subtraction_with_negative_bound(self, a, b):
        c = a - b
        s = ArbitraryPrecisionBound(a.lower_bound, None)
        assert c in s


class TestMultiplication(object):
    @given(
        arbitrary_precision_bounds(),
    )
    def test_multiplication_with_zero(self, a):
        assert (a * 0).is_zero()

    @given(
        arbitrary_precision_bounds(),
    )
    def test_multiplication_with_one(self, a):
        assert a * 1.0 == a

    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_commutative_property(self, a, b):
        c = a * b
        d = b * a
        assert c == d

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(lower_bound=0),
    )
    def test_multiplication_positive_with_positive(self, a, b):
        c = a * b
        assert c.is_nonnegative()

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(upper_bound=0),
    )
    def test_multiplication_positive_with_negative(self, a, b):
        c = a * b
        assert c.is_nonpositive()


class TestDivision(object):
    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_division_by_zero(self, a, b):
        assume(b.lower_bound < 0 and b.upper_bound > 0)
        c = a / b
        assert c == ArbitraryPrecisionBound(None, None)

    @given(
        arbitrary_precision_bounds(),
    )
    def test_division_by_one(self, a):
        assert a / 1.0 == a

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(lower_bound=1),
    )
    def test_division_positive_with_positive(self, a, b):
        c = a / b
        assert c.is_nonnegative()

    @given(
        arbitrary_precision_bounds(lower_bound=0),
        arbitrary_precision_bounds(upper_bound=-1),
    )
    def test_division_positive_with_negative(self, a, b):
        c = a / b
        assert c.is_nonpositive()


class TestContains(object):
    @given(
        arbitrary_precision_bounds(),
    )
    def test_bound_contains_itself(self, a):
        assert a in a

    @given(
        arbitrary_precision_bounds(allow_infinity=False),
    )
    def test_bound_contains_midpoint(self, a):
        m = (a.lower_bound + a.upper_bound) / 2.0
        assert m in a

    @given(reals())
    def test_infinity_bound_contains_everything(self, n):
        a = ArbitraryPrecisionBound(None, None)
        assert n in a


class TestTighten(object):
    @given(
        arbitrary_precision_bounds(),
        arbitrary_precision_bounds(),
    )
    def test_bound_is_tighter(self, a, b):
        assume(a.upper_bound > b.lower_bound)
        assume(a.lower_bound < b.upper_bound)
        c = a.tighten(b)
        assert c.lower_bound >= min(a.lower_bound, b.lower_bound)
        assert c.upper_bound <= max(a.upper_bound, b.upper_bound)

# pylint: skip-file
import pytest
from hypothesis import given, assume
import hypothesis.strategies as st
from tests.strategies import reals
from suspect.interval import Interval, EmptyInterval
from suspect.math import *


@st.composite
def intervals(draw, allow_infinity=True, lower_bound=None, upper_bound=None):
    if lower_bound is None and allow_infinity:
        lower = draw(st.one_of(
            st.none(),
            reals(max_value=upper_bound, allow_infinity=False)
        ))
    else:
        lower = draw(reals(
            min_value=lower_bound,
            max_value=upper_bound,
            allow_infinity=False
        ))

    if upper_bound is None and allow_infinity:
        upper = draw(st.one_of(
            st.none(),
            reals(min_value=lower, allow_infinity=False)
        ))
    else:
        upper = draw(reals(
            min_value=lower,
            max_value=upper_bound,
            allow_infinity=False,
        ))
    return Interval(lower, upper)


class TestAddition(object):
    @given(intervals())
    def test_addition_with_zero(self, bound):
        zero_bound = Interval.zero()
        assert bound + zero_bound == bound

    @given(intervals(), intervals())
    def test_commutative_property(self, a, b):
        assert a + b == b + a

    @given(intervals(), intervals(lower_bound=0))
    def test_addition_with_positive_bound(self, a, b):
        c = a + b
        s = Interval(a.lower_bound, None)
        assert c in s

    @given(intervals(), intervals(upper_bound=0))
    def test_addition_with_negative_bound(self, a, b):
        c = a + b
        s = Interval(None, a.upper_bound)
        assert c in s


class TestSubtraction(object):
    @given(intervals())
    def test_negation(self, a):
        assert (-(-a)) == a

    @given(intervals())
    def test_subtraction_with_zero(self, a):
        assert a - 0 == a

    @given(
        intervals(),
        intervals(lower_bound=0),
    )
    def test_subtraction_with_positive_bound(self, a, b):
        c = a - b
        s = Interval(None, a.upper_bound)
        assert c in s

    @given(
        intervals(),
        intervals(upper_bound=0),
    )
    def test_subtraction_with_negative_bound(self, a, b):
        c = a - b
        s = Interval(a.lower_bound, None)
        assert c in s


class TestMultiplication(object):
    @given(intervals())
    def test_multiplication_with_zero(self, a):
        assert (a * 0).is_zero()

    @given(intervals())
    def test_multiplication_with_one(self, a):
        assert a * 1.0 == a

    @given(intervals(), intervals())
    def test_commutative_property(self, a, b):
        c = a * b
        d = b * a
        assert c == d

    @given(intervals(lower_bound=0), intervals(lower_bound=0))
    def test_multiplication_positive_with_positive(self, a, b):
        c = a * b
        assert c.is_nonnegative()

    @given(intervals(lower_bound=0), intervals(upper_bound=0))
    def test_multiplication_positive_with_negative(self, a, b):
        c = a * b
        assert c.is_nonpositive()


class TestDivision(object):
    @given(intervals(), intervals())
    def test_division_by_zero(self, a, b):
        assume(b.lower_bound < 0 and b.upper_bound > 0)
        c = a / b
        assert c == Interval(None, None)

    @given(intervals())
    def test_division_by_one(self, a):
        assert a / 1.0 == a

    @given(intervals(lower_bound=0), intervals(lower_bound=1))
    def test_division_positive_with_positive(self, a, b):
        c = a / b
        assert c.is_nonnegative()

    @given(intervals(lower_bound=0), intervals(upper_bound=-1))
    def test_division_positive_with_negative(self, a, b):
        c = a / b
        assert c.is_nonpositive()


class TestContains(object):
    @given(intervals())
    def test_bound_contains_itself(self, a):
        assert a in a

    @given(intervals(allow_infinity=False))
    def test_bound_contains_midpoint(self, a):
        assume((a.upper_bound - a.lower_bound) != inf)
        m = a.lower_bound + (a.upper_bound - a.lower_bound) / 2.0
        assert m in a

    @given(reals())
    def test_infinity_bound_contains_everything(self, n):
        a = Interval(None, None)
        assert n in a


class TestIntersect(object):
    @given(intervals(), intervals())
    def test_bound_is_tighter(self, a, b):
        assume(a.upper_bound > b.lower_bound)
        assume(a.lower_bound < b.upper_bound)
        c = a.intersect(b)
        assert c.lower_bound >= min(a.lower_bound, b.lower_bound)
        assert c.upper_bound <= max(a.upper_bound, b.upper_bound)


class TestSize(object):
    @given(reals())
    def test_infinite_lower_bound_greater_than_everything(self, f):
        b = Interval(None, 0)
        assert almostgte(b.size(), f)

    @given(reals())
    def test_infinite_upper_bound_greater_than_everything(self, f):
        b = Interval(0, None)
        assert almostgte(b.size(), f)

    @given(reals())
    def test_infinite_bounds_greater_than_everything(self, f):
        b = Interval(None, None)
        assert almostgte(b.size(), f)

    @given(
        intervals(allow_infinity=False, upper_bound=1e10),
        reals(allow_infinity=False, min_value=1.0, max_value=1e20)
    )
    def test_finite_bounds(self, a, f):
        b = a * f
        assert almostlte(a.size(), b.size())

    @given(
        intervals(allow_infinity=False),
        reals(allow_infinity=False, min_value=0.0, max_value=1.0)
    )
    def test_finite_bounds_1(self, a, f):
        b = a * f
        assert almostgte(a.size(), b.size())


class TestAbs(object):
    @pytest.mark.parametrize('it,expected', [
        (Interval(1, 2), Interval(1, 2)),
        (Interval(-3, -2), Interval(2, 3)),
        (Interval(-3, 2), Interval(0, 3)),
        (Interval(-2, 3), Interval(0, 3)),
    ])
    def test_abs(self, it, expected):
        assert expected == abs(it)


class TestPower(object):
    @pytest.mark.parametrize('base,p,expected', [
        (Interval(2, 3), 3, Interval(8, 27)),
        (Interval(2, 3), 4, Interval(16, 81)),
        (Interval(-3, 2), 3, Interval(-27, 8)),
        (Interval(-3, 2), 4, Interval(0, 81)),
        (Interval(-3, -2), 3, Interval(-27, -8)),
        (Interval(-3, -2), 4, Interval(16, 81)),

        (Interval(2, 4), -3, Interval(1.0/64.0, 1.0/8.0)),
        (Interval(2, 4), -4, Interval(1.0/256.0, 1.0/16.0)),
        (Interval(-4, -2), -3, Interval(-1.0/8.0, -1.0/64.0)),
        (Interval(-4, -2), -4, Interval(1.0/256.0, 1.0/16.0)),

        (Interval(2, 3), 0, Interval(1, 1)),
        (Interval(-3, 2), 0, Interval(1, 1)),
        (Interval(-3, -2), 0, Interval(1, 1)),
    ])
    def test_power(self, base, p, expected):
        result = base ** p
        assert expected == result

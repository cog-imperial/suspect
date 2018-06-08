# pylint: skip-file
import pytest
from suspect.math import RoundMode as RM
from suspect.math.arbitrary_precision import *


@pytest.mark.parametrize('func', [sqrt, log, exp, sin])
def test_func(func):
    assert func(make_number(123.4), RM.RN) == func(make_number(123.4), RM.RD)
    assert func(make_number(123.4), RM.RU) == func(make_number(123.4), RM.RD)
    assert func(make_number(123.4), RM.RZ) == func(make_number(123.4), RM.RD)


class TestAlmostEq(object):
    def test_inf(self):
        assert almosteq(inf, inf)
        assert not almosteq(inf, -inf)
        assert almosteq(-inf, -inf)

    def test_finite(self):
        assert almosteq(pi, pi)
        assert not almosteq(2e20, inf)


class TestAlomstGte(object):
    def test_inf(self):
        assert almostgte(inf, -inf)
        assert almostgte(inf, inf)
        assert almostgte(-inf, -inf)

    def test_finite(self):
        assert almostgte(pi, pi)


class TestAlomstLte(object):
    def test_inf(self):
        assert almostlte(-inf, inf)
        assert almostlte(inf, inf)
        assert almostlte(-inf, -inf)

    def test_finite(self):
        assert almostlte(pi, pi)

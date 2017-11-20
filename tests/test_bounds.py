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
from mpmath import mpf, mp
import mpmath
import pyomo.environ as aml
from convexity_detection.expr_visitor import (
    bottom_up_visit as visit_expression
)
from convexity_detection.bounds import (
    Bound,
    BoundsHandler,
    expression_bounds,
    is_positive,
    is_nonnegative,
)
from convexity_detection.math import inf
from util import _var

pi = mpmath.pi


def test_bounds_arithmetic():
    a = Bound('2', '3')
    b = Bound('-6', '-3')

    # sanity check
    assert a != b

    # [a, b] + [c, d] = [a + c, b + d]
    assert (a + b) == Bound('-4', '0')
    assert (mpf('6.2') + b) == Bound('0.2', '3.2')
    assert (a + 0) == a
    # [a, b] - [c, d] = [a - d, b - c]
    assert (a - b) == Bound('5', '9')
    assert (a - 0) == a
    # [a, b] * [c, d] = [min(ac, ad, bc, bd), max(ac, ad, bc, bd)]
    assert (a * b) == Bound('-18', '-6')
    assert (a * 0) == Bound('0', '0')
    # [a, b] / [c, d] = [a, b] * [d^-1, c^-1] if a, b !=0 else [-inf, inf]
    assert (a / 1) == a
    assert (a / b) == Bound('-1', -mpf('1')/mpf('3'))
    assert (a / 0) == Bound(-inf, inf)

    assert Bound('0.2', '1') in Bound(-1, 1)
    assert Bound('-1', '1') in Bound(-1, 1)
    assert Bound('-2', '1') not in Bound(-1, 1)


def test_bound_tighten():
    a = Bound('-1', '2')
    b = Bound('0', '3')
    assert a.tighten(b) == Bound('0', '2')
    assert a.tighten(b) == b.tighten(a)

    c = Bound('1', '1.2')
    assert b.tighten(c) == Bound('1', '1.2')


def test_bounds_handler_tightening():
    handler = BoundsHandler()

    v0 = _var()
    e0 = v0 + 2
    visit_expression(handler, e0)
    assert handler.bound(e0) == Bound(-inf, inf)

    v0.setlb(10)
    v0.setub(20)
    e1 = v0 + 2
    visit_expression(handler, e1)
    assert handler.bound(e0) == Bound(12, 22)


def test_bound_simple_var():
    assert expression_bounds(_var()) == Bound(None, None)
    assert expression_bounds(_var((None, -3))) == Bound(None, -3)
    assert expression_bounds(_var((-8, 5))) == Bound(-8, 5)


def test_bound_product():
    assert expression_bounds(10*_var()) == Bound(None, None)
    assert expression_bounds(10*_var((-1, 1))) == Bound(-10, 10)
    assert expression_bounds(-10*_var((-2, 1))) == Bound(-10, 20)


def test_bound_linear():
    #    [-12, 18]         + [0, inf]            - [10, 20]
    e0 = 6 * _var((-2, 3)) + 2 * _var((0, None)) - 10 * _var((1, 2))
    assert expression_bounds(e0) == Bound(-32, None)

    #    ([-2, 3]       + [0, inf])            * [2, 4]
    e1 = (_var((-2, 3)) + _var((0, None))) * 2 * _var((1, 2))
    #  = [-2, inf] * [2, 4]

    assert expression_bounds(e1) == Bound(-8, None)
    assert expression_bounds(3 - _var((0, 2))) == Bound(1, 3)


def test_floating_point_erasure():
    def _test_floating_erasure():
        bound = expression_bounds(mpf('1e16') * _var((1, 2))
                            + mpf('1e-16') * _var((1, 2)))
        assert bound.l > 1e16
        assert bound.u > 2e16

    # with default precision this test fails
    with pytest.raises(AssertionError):
        _test_floating_erasure()

    # with higher precision it works
    with mpmath.workdps(50):
        _test_floating_erasure()


def test_bound_abs():
    e0 = abs(_var((-10, -1)))
    assert expression_bounds(e0) == Bound(1, 10)

    e1 = abs(_var((1, 10)))
    assert expression_bounds(e1) == Bound(1, 10)

    e2 = abs(_var((-10, 20)))
    assert expression_bounds(e2) == Bound(0, 20)


def test_bound_sqrt():
    with pytest.raises(ValueError):
        expression_bounds(aml.sqrt(_var()))
    assert expression_bounds(aml.sqrt(_var((0, None)))) == Bound(0, None)
    assert expression_bounds(aml.sqrt(_var((0, 2)))) == Bound(0, mpmath.sqrt(2))


def test_bound_log():
    with pytest.raises(ValueError):
        expression_bounds(aml.log(_var((0, None))))
    assert expression_bounds(aml.log(_var((1, None)))) == Bound(0, None)
    assert expression_bounds(aml.log(_var((1, 2)))) == Bound(0, mpmath.log(2))


def test_bound_asin():
    with pytest.raises(ValueError):
        expression_bounds(aml.asin(_var((0, None))))
    assert expression_bounds(aml.asin(_var((-1, 1)))) == Bound(-pi/2, pi/2)


def test_bound_acos():
    with pytest.raises(ValueError):
        expression_bounds(aml.acos(_var((0, None))))
    assert expression_bounds(aml.acos(_var((-1, 1)))) == Bound(0, pi)


def test_bound_atan():
    assert expression_bounds(aml.atan(_var())) == Bound(-pi/2, pi/2)


def test_bound_exp():
    assert expression_bounds(aml.exp(_var())) == Bound(0, None)
    assert expression_bounds(aml.exp(_var((0, None)))) == Bound(1, None)


def test_bound_sin():
    def assert_sin_bound(bounds, expected_bounds):
        assert expression_bounds(aml.sin(_var(bounds))) == Bound(*expected_bounds)

    assert_sin_bound((None, None), (-1, 1))
    assert_sin_bound((0, mpf(pi)), (0, 1))
    assert_sin_bound((-pi, 0), (-1, 0))
    assert_sin_bound(
        (0.5 * pi - 0.5, 0.5 * pi + 0.5),
        (mpmath.sin(0.5 * pi - 0.5), 1)
    )
    assert_sin_bound(
        (pi - 0.5, pi + 0.5),
        (mpmath.sin(pi + 0.5), mpmath.sin(pi - 0.5))
    )
    assert_sin_bound(
        (2 * pi - 0.1, 2 * pi + 0.5),
        (mpmath.sin(2*pi - 0.1), mpmath.sin(2*pi + 0.5))
    )


def test_bound_cos():
    def assert_cos_bound(bounds, expected_bounds):
        assert expression_bounds(aml.cos(_var(bounds))) == Bound(*expected_bounds)

    assert_cos_bound((None, None), (-1, 1))
    assert_cos_bound((0, mpf(pi)), (-1, 1))
    assert_cos_bound((-0.5*pi, 0.5*pi), (0, 1))
    assert_cos_bound((0, 0.5*pi), (0, 1))
    assert_cos_bound((pi - 0.2, 2*pi), (-1, 1))
    assert is_nonnegative(aml.cos(_var((mpf('1.5')*mpf(pi), mpf('2.5')*mpf(pi)))))


def test_bound_tan():
    def assert_tan_bound(bounds, expected_bounds):
        assert expression_bounds(aml.tan(_var(bounds))) == Bound(*expected_bounds)

    assert_tan_bound((None, None), (None, None))
    assert_tan_bound((0, 0.5 * pi), (0, None))
    assert_tan_bound((0.5*pi, 0.5*pi), (None, None))
    assert_tan_bound((pi-0.1, pi+0.1), (-mpmath.tan(0.1), mpmath.tan(0.1)))
    assert_tan_bound(
        (0.5*pi-0.1, 0.5*pi+0.1),
        (mpmath.tan(0.5*pi+0.1), mpmath.tan(0.5*pi-0.1))
    )


def test_bound_negation():
    e0 = -_var((-10, 1))
    assert expression_bounds(e0) == Bound(-1, 10)

    e1 = -abs(_var())
    assert expression_bounds(e1) == Bound(None, 0)


def test_is_positive():
    e0 = _var() + 3
    assert not is_positive(e0)

    e1 = abs(_var((-8, -3)))
    assert is_positive(e1)

    e2 = abs(_var() * 3.0 - 20)
    assert is_nonnegative(e2)

    assert is_nonnegative(aml.sin(_var((0, float(pi)))))
    # assert is_nonpositive(aml.cos(_var((float(0.5*pi), float(1.5*pi)))))


def test_is_negative():
    assert not Bound('-1', '2').is_negative()
    assert not Bound(None, 0).is_negative()
    assert Bound('-1', '-0.2').is_negative()

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
import pyomo.environ as aml
from suspect.convexity import *
from suspect.math import pi
from util import _var


def test_convexity_enum():
    assert Convexity.Convex.is_convex()
    assert not Convexity.Convex.is_concave()
    assert not Convexity.Convex.is_linear()
    assert Convexity.Concave.is_concave()
    assert not Convexity.Concave.is_convex()
    assert not Convexity.Concave.is_linear()
    assert Convexity.Linear.is_linear()
    assert Convexity.Linear.is_convex()
    assert Convexity.Linear.is_concave()


def test_const_is_linear():
    assert expression_convexity(-2.0).is_linear()


def test_var_is_linear():
    x = _var()
    assert expression_convexity(x).is_linear()


def test_convexity_linear():
    assert expression_convexity(
        sum(i*_var() for i in range(1, 20))
    ).is_linear()


def test_convexity_sqrt():
    with pytest.raises(ValueError):
        assert expression_convexity(aml.sqrt(_var()))
    assert expression_convexity(aml.sqrt(_var((-10, None)) + 10)).is_concave()
    assert expression_convexity(aml.sqrt(aml.exp(_var()))).is_unknown()


def test_convexity_exp():
    assert expression_convexity(aml.exp(_var())) == Convexity.Convex
    assert expression_convexity(aml.exp(-_var())) == Convexity.Convex
    assert expression_convexity(aml.exp(0)) == Convexity.Linear


def test_convexity_log():
    with pytest.raises(ValueError):
        assert expression_convexity(aml.log(_var()))

    assert expression_convexity(aml.log(_var((1, None)))).is_concave()
    assert expression_convexity(aml.log(aml.sqrt(1+_var((0, None))))).is_concave()
    # assert expression_convexity(aml.log(aml.exp(_var()))).is_linear()


def test_convexity_abs():
    # linear case
    assert expression_convexity(abs(-_var() + 2)) == Convexity.Convex
    assert expression_convexity(abs(0.0)).is_linear()

    # g concave, g >= 0
    assert expression_convexity(abs(aml.sqrt(_var((0, None))))) == Convexity.Concave
    # g convex, g <= 0
    assert expression_convexity(abs(-aml.sqrt(_var((0, None))))) == Convexity.Concave

    # g concave, g <= 0
    assert expression_convexity(abs(-aml.exp(_var()))) == Convexity.Convex
    # g convex, g >= 0
    assert expression_convexity(abs(aml.exp(_var()))) == Convexity.Convex


def test_convexity_sin():
    def assert_sin(l, u, expected):
        assert expression_convexity(aml.sin(_var((l, u)))) == expected

    assert_sin(None, None, Convexity.Unknown)
    assert_sin(-0.5, 0.5, Convexity.Unknown)
    assert_sin(0, 0.5*pi, Convexity.Concave)
    assert_sin(2*pi, 3*pi, Convexity.Concave)
    assert_sin(3*pi, 4*pi, Convexity.Convex)


def test_convexity_cos():
    def assert_cos(l, u, expected):
        assert expression_convexity(aml.cos(_var((l, u)))) == expected

    assert_cos(None, None, Convexity.Unknown)
    assert_cos(0.5*pi-0.2, 0.5*pi+0.3, Convexity.Unknown)
    assert_cos(-0.5*pi, 0.5*pi, Convexity.Concave)
    assert_cos(0.5*pi, 1.5*pi, Convexity.Convex)
    assert_cos(1.5*pi, 2.5*pi, Convexity.Concave)


def test_convexity_tan():
    def assert_tan(l, u, expected):
        assert expression_convexity(aml.tan(_var((l, u)))) == expected

    assert_tan(0, 2*pi, Convexity.Unknown)
    assert_tan(-0.5, 0.5, Convexity.Unknown)
    assert_tan(0.5*pi-0.1, 0.5*pi+0.1, Convexity.Unknown)
    assert_tan(0, 0.5*pi, Convexity.Convex)
    assert_tan(-0.5*pi, 0, Convexity.Concave)

    assert expression_convexity(
        aml.tan(aml.exp(_var((None, 0))))
    ) == Convexity.Convex

    assert expression_convexity(
        aml.tan(aml.sqrt(_var((0, None))))
    ) == Convexity.Unknown

    assert expression_convexity(
        aml.tan(-aml.exp(_var((None, 0))))
    ) == Convexity.Concave

    assert expression_convexity(
        aml.tan(-aml.sqrt(_var((0, None))))
    ) == Convexity.Unknown


def test_convexity_asin():
    def assert_asin(l, u, expected):
        assert expression_convexity(aml.asin(_var((l, u)))) == expected

    assert_asin(-1, 1, Convexity.Unknown)
    with pytest.raises(ValueError):
        assert_asin(-2, 0, Convexity.Unknown)
    assert_asin(-1, 0, Convexity.Concave)
    assert_asin(0, 1, Convexity.Convex)

    # convex, [0, 1]
    assert expression_convexity(
        aml.asin(aml.exp(_var((None, 0))))
    ) == Convexity.Convex

    # concave, [0, 1]
    assert expression_convexity(
        aml.asin(aml.sqrt(_var((0, 1))))
    ) == Convexity.Unknown

    # concave, [-1, 0]
    assert expression_convexity(
        aml.asin(-aml.exp(_var((None, 0))))
    ) == Convexity.Concave

    # convex, [-1, 0]
    assert expression_convexity(
        aml.asin(aml.exp(_var((None, 0)))-1)
    ) == Convexity.Unknown


def test_convexity_acos():
    def assert_acos(l, u, expected):
        assert expression_convexity(aml.acos(_var((l, u)))) == expected

    assert_acos(-1, 1, Convexity.Unknown)
    with pytest.raises(ValueError):
        assert_acos(-2, 0, Convexity.Unknown)
    assert_acos(-1, 0, Convexity.Convex)
    assert_acos(0, 1, Convexity.Concave)

    # convex, [0, 1]
    assert expression_convexity(
        aml.acos(aml.exp(_var((None, 0))))
    ) == Convexity.Concave

    # concave, [0, 1]
    assert expression_convexity(
        aml.acos(aml.sqrt(_var((0, 1))))
    ) == Convexity.Unknown

    # concave, [-1, 0]
    assert expression_convexity(
        aml.acos(-aml.exp(_var((None, 0))))
    ) == Convexity.Convex

    # convex, [-1, 0]
    assert expression_convexity(
        aml.acos(aml.exp(_var((None, 0)))-1)
    ) == Convexity.Unknown


def test_convexity_atan():
    assert expression_convexity(aml.atan(_var((0, None)))) == Convexity.Concave
    assert expression_convexity(aml.atan(_var((None, 0)))) == Convexity.Convex
    assert expression_convexity(aml.atan(_var())) == Convexity.Unknown


    # convex, [0, 1]
    assert expression_convexity(
        aml.atan(aml.exp(_var((None, 0))))
    ) == Convexity.Unknown

    # concave, [0, 1]
    assert expression_convexity(
        aml.atan(aml.sqrt(_var((0, 1))))
    ) == Convexity.Concave

    # concave, [-1, 0]
    assert expression_convexity(
        aml.atan(-aml.exp(_var((None, 0))))
    ) == Convexity.Unknown

    # convex, [-1, 0]
    assert expression_convexity(
        aml.atan(aml.exp(_var((None, 0)))-1)
    ) == Convexity.Convex

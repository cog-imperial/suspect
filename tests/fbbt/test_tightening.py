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

# pylint: skip-file
import pytest
from unittest.mock import MagicMock
from hypothesis import given, assume
import hypothesis.strategies as st
from suspect.fbbt.tightening import *
import suspect.dag.expressions as dex
from suspect.interval import Interval
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
)

@pytest.fixture
def visitor():
    return BoundsTighteningVisitor()


class TestPower(object):
    def test_square(self, visitor, ctx):
        c = dex.Constant(2.0)
        p = PlaceholderExpression()
        pow_ = dex.PowExpression(children=[p, c])
        bounds = {}
        bounds[pow_] = Interval(0, 4)
        visitor.visit(pow_, bounds)
        assert bounds[p] == Interval(-2, 2)

    def test_non_square(self, visitor, ctx):
        c = dex.Constant(3.0)
        p = PlaceholderExpression()
        bounds = {}
        bounds[p] = Interval(None, None)
        pow_ = dex.PowExpression(children=[p, c])
        bounds[pow_] = Interval(0, 4)
        visitor.visit(pow_, bounds)
        assert bounds[p] == Interval(None, None)

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
from tests.conftest import PlaceholderExpression
from suspect.dag.dag import VerticesList
from hypothesis import given, assume
import hypothesis.strategies as st


class TestVerticesList(object):
    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_with_starting_vertices(self, depths, reverse):
        vertices = [PlaceholderExpression() for _ in depths]
        for i, v in enumerate(vertices):
            v.depth = depths[i]
        vl = VerticesList(vertices, reverse=reverse)
        assert [v.depth for v in vl] == sorted(depths, reverse=reverse)

    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_append(self, depths, reverse):
        vl = VerticesList(reverse=reverse)
        for d in depths:
            p = PlaceholderExpression()
            p.depth = d
            vl.append(p)
        assert [v.depth for v in vl] == sorted(depths, reverse=reverse)

    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_pop(self, depths, reverse):
        vertices = [PlaceholderExpression() for _ in depths]
        for i, v in enumerate(vertices):
            v.depth = depths[i]
        vl = VerticesList(vertices, reverse=reverse)
        if reverse:
            assert vl.pop().depth == max(depths)
        else:
            assert vl.pop().depth == min(depths)

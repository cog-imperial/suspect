# pylint: skip-file
import pytest
from tests.conftest import PlaceholderExpression
from suspect.dag.dag import VerticesList
from hypothesis import given, assume
import hypothesis.strategies as st


class TestVerticesList(object):
    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_append(self, depths, reverse):
        vl = VerticesList(reverse=reverse)
        for d in depths:
            p = PlaceholderExpression()
            vl.append(p, d)
        assert vl._vertices_depth == sorted(depths, reverse=reverse)

    @pytest.mark.parametrize('reverse', [True, False])
    @given(depths=st.lists(st.integers(min_value=0, max_value=100), min_size=1))
    def test_pop(self, depths, reverse):
        vertices = [PlaceholderExpression() for _ in depths]
        depths_map = {}
        vl = VerticesList(reverse=reverse)
        for i, v in enumerate(vertices):
            depths_map[id(v)] = depths[i]
            vl.append(v, depths[i])

        popped = vl.pop()
        if reverse:
            assert depths_map[id(popped)] == max(depths)
        else:
            assert depths_map[id(popped)] == min(depths)

    def test_bool(self):
        vl = VerticesList()
        assert not vl
        vl.append(PlaceholderExpression(), 1)
        assert vl
        vl.pop()
        assert not vl

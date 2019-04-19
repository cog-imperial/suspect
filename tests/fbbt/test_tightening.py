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
import pyomo.environ as pe
from pyomo.core.kernel.component_map import ComponentMap
import suspect.pyomo.expressions as dex
import numpy as np
from suspect.interval import Interval as I
from suspect.fbbt.tightening.rules import *
from suspect.fbbt.tightening.visitor import BoundsTighteningVisitor
from suspect.expression import UnaryFunctionType as UFT
from tests.strategies import coefficients, reals
from tests.conftest import (
    bound_description_to_bound,
    mono_description_to_mono,
)


@pytest.fixture
def visitor():
    return BoundsTighteningVisitor()


@given(reals(), reals())
def test_constraint_rule(visitor, bound1, bound2):
    lower_bound = min(bound1, bound2)
    upper_bound = max(bound1, bound2)
    rule = ConstraintRule()
    child = pe.Var()
    constraint = dex.Constraint(
        'test', lower_bound=lower_bound, upper_bound=upper_bound, children=[child]
    )
    matched, result = visitor.visit_expression(constraint, None)
    assert matched
    assert result[0] == I(lower_bound, upper_bound)


class TestSumRule:
    children = None
    expr = None

    def _rule_result(self, visitor, children_bounds, expr_bounds):
        children = [pe.Var() for _ in children_bounds]
        expr = sum(children)

        bounds = ComponentMap()
        bounds[expr] = expr_bounds
        for child, child_bounds in zip(children, children_bounds):
            bounds[child] = child_bounds

        self.children = children
        self.expr = expr
        matched, result = visitor.visit_expression(expr, bounds)
        assert matched
        return result

    def test_unbounded_expr(self, visitor):
        new_bounds = self._rule_result(visitor, [I(0, 1), I(-1, 0)], I(0, None))
        assert new_bounds[0] == I(0, None)
        assert new_bounds[1] == I(-1, None)

    def test_bounded_expr(self, visitor):
        expr_bounds = I(-5 , 5)
        children_bounds = [I(1, 2), I(-2, 2), I(-10, 10)]
        new_bounds = self._rule_result(visitor, children_bounds, expr_bounds)
        assert new_bounds[2] == I(-9, 6)


@pytest.mark.skip('Linear Expression Tag not implemented')
class TestLinearRule:
    children = None
    expr = None

    def _rule_result(self, coefficients, children_bounds, expr_bounds):
        children = [PE() for _ in children_bounds]
        expr = PE(ET.Linear, children, coefficients=coefficients, constant_term=0)

        bounds = ComponentMap()
        bounds[expr] = expr_bounds
        for child, child_bounds in zip(children, children_bounds):
            bounds[child] = child_bounds

        rule = LinearRule()
        self.children = children
        self.expr = expr
        return rule.apply(expr, bounds)

    def test_unbounded_expr(self):
        new_bounds = self._rule_result([-1, 1], [I(0, 1), I(-1, 0)], I(0, None))
        assert new_bounds[0] == I(None, 0)
        assert new_bounds[1] == I(0, None)

    def test_bounded_expr(self):
        expr_bounds = I(-5 , 5)
        children_bounds = [I(-2, -1), I(-2, 2), I(-10, 10)]
        new_bounds = self._rule_result([-1, 1, 2], children_bounds, expr_bounds)
        assert new_bounds[2] == I(-4.5, 3)


@pytest.mark.skip('Tagged quadratic expressions are not implemented')
class TestQuadraticRule:
    children = None
    expr = None

    def _rule_result(self, terms, children, children_bounds, expr_bounds):
        expr = PE(ET.Quadratic, children, terms=terms)

        bounds = ComponentMap()
        bounds[expr] = expr_bounds
        for child, child_bounds in zip(children, children_bounds):
            bounds[child] = child_bounds

        rule = QuadraticRule()
        self.children = children
        self.expr = expr
        return rule.apply(expr, bounds)

    def test_sum_of_squares(self):
        x = PE()
        y = PE()
        new_bounds = self._rule_result(
            [BilinearTerm(x, x, 1.0), BilinearTerm(y, y, 2.0)],
            [x, y],
            [I(None, None), I(None, None)],
            I(None, 10),
        )
        assert new_bounds[0] == I(-np.sqrt(10.0), np.sqrt(10))
        assert new_bounds[1] == I(-np.sqrt(5), np.sqrt(5))


class TestPowerRuleConstantExpo:
    base = None

    def _rule_result(self, visitor, expo, expr_bounds):
        base = pe.Var()
        expr = base ** expo
        self.base = base

        bounds = ComponentMap()
        bounds[expr] = expr_bounds
        return visitor.visit_expression(expr, bounds)

    def test_square(self, visitor):
        matched, new_bounds = self._rule_result(visitor, 2.0, I(0, 4))
        assert matched
        assert new_bounds[0] == I(-2, 2)

    def test_non_square(self, visitor):
        matched, new_bounds = self._rule_result(visitor, 3.0, I(0, 4))
        assert not matched
        assert new_bounds is None


class TestUnboundedFunctionsRule:
    @pytest.mark.parametrize('expr_type', [
        pe.sin, pe.cos, pe.tan, pe.asin, pe.acos, pe.atan,
    ])
    @given(a=reals(), b=reals())
    def test_unbounded(self, visitor, a, b, expr_type):
        lower = min(a, b)
        upper = max(a, b)
        child = pe.Var(bounds=(lower, upper))
        expr = expr_type(child)
        bounds = ComponentMap()
        bounds[expr] = I(lower, upper)
        matched, result = visitor.visit_expression(expr, bounds)
        assert matched
        assert result[0] == I(None, None)


class TestVisitor:
    def test_handle_result_with_new_value(self, visitor):
        bounds = ComponentMap()
        expr = pe.Var()
        assert visitor.handle_result(expr, Interval(0, 1), bounds)
        assert bounds[expr] == Interval(0, 1)

    def test_handle_result_with_no_new_value(self, visitor):
        bounds = ComponentMap()
        expr = pe.Var()
        visitor.handle_result(expr, Interval(0, 1), bounds)
        assert not visitor.handle_result(expr, None, bounds)
        assert bounds[expr] == Interval(0, 1)

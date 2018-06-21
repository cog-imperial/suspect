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
import suspect.dag.expressions as dex
from suspect.interval import Interval as I
from suspect.expression import ExpressionType as ET, UnaryFunctionType as UFT
from suspect.fbbt.tightening.rules import *
from tests.conftest import (
    PlaceholderExpression as PE,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
)


class TighteningContext:
    def __init__(self, bounds=None):
        if bounds is None:
            bounds = {}
        self._bounds = bounds

    def bounds(self, expr):
        return self._bounds[expr]


@given(reals(), reals())
def test_constraint_rule(bound1, bound2):
    lower_bound = min(bound1, bound2)
    upper_bound = max(bound1, bound2)
    rule = ConstraintRule()
    ctx = TighteningContext()
    child = PE()
    result = rule.checked_apply(
        PE(ET.Constraint, [child], lower_bound=lower_bound, upper_bound=upper_bound),
        ctx,
    )
    assert result[child] == I(lower_bound, upper_bound)


class TestSumRule:
    children = None
    expr = None

    def _rule_result(self, children_bounds, expr_bounds):
        children = [PE() for _ in children_bounds]
        expr = PE(ET.Sum, children)

        starting_bounds = {expr: expr_bounds}
        for child, bounds in zip(children, children_bounds):
            starting_bounds[child] = bounds
        ctx = TighteningContext(bounds=starting_bounds)
        rule = SumRule()
        self.children = children
        self.expr = expr
        return rule.checked_apply(expr, ctx)

    def test_unbounded_expr(self):
        new_bounds = self._rule_result([I(0, 1), I(-1, 0)], I(0, None))
        assert new_bounds is None

    def test_bounded_expr(self):
        expr_bounds = I(-5 , 5)
        children_bounds = [I(1, 2), I(-2, 2), I(-10, 10)]
        new_bounds = self._rule_result(children_bounds, expr_bounds)
        assert new_bounds[self.children[2]] == I(-9, 6)


class TestLinearRule:
    children = None
    expr = None

    def _rule_result(self, coefficients, children_bounds, expr_bounds):
        children = [PE() for _ in children_bounds]
        expr = PE(ET.Linear, children, coefficients=coefficients, constant_term=0)

        starting_bounds = {expr: expr_bounds}
        for child, bounds in zip(children, children_bounds):
            starting_bounds[child] = bounds
        ctx = TighteningContext(bounds=starting_bounds)
        rule = LinearRule()
        self.children = children
        self.expr = expr
        return rule.checked_apply(expr, ctx)

    def test_unbounded_expr(self):
        new_bounds = self._rule_result([-1, 1], [I(0, 1), I(-1, 0)], I(0, None))
        assert new_bounds is None

    def test_bounded_expr(self):
        expr_bounds = I(-5 , 5)
        children_bounds = [I(-2, -1), I(-2, 2), I(-10, 10)]
        new_bounds = self._rule_result([-1, 1, 2], children_bounds, expr_bounds)
        assert new_bounds[self.children[2]] == I(-4.5, 3)


class TestPowerRuleConstantExpo:
    base = None

    def _rule_result(self, expo, expr_bounds):
        base = PE()
        expo = PE(ET.Constant, value=expo, is_constant=True)
        expr = PE(ET.Power, [base, expo])
        rule = PowerRule()
        self.base = base
        ctx = TighteningContext(bounds={expr: expr_bounds})
        return rule.checked_apply(expr, ctx)

    def test_square(self):
        new_bounds = self._rule_result(2.0, I(0, 4))
        assert new_bounds[self.base] == I(-2, 2)

    def test_non_square(self):
        new_bounds = self._rule_result(3.0, I(0, 4))
        assert new_bounds is None


class TestUnboundedFunctionsRule:
    @pytest.mark.parametrize('expr_type,rule_cls', [
        (UFT.Sin, SinRule), (UFT.Cos, CosRule), (UFT.Tan, TanRule),
        (UFT.Asin, AsinRule), (UFT.Acos, AcosRule), (UFT.Atan, AtanRule)
    ])
    @given(a=reals(), b=reals())
    def test_unbounded(self, a, b, expr_type, rule_cls):
        lower = min(a, b)
        upper = max(a, b)
        rule = rule_cls()
        child = PE()
        expr = PE(ET.UnaryFunction, [child], func_type=expr_type)
        ctx = TighteningContext({expr: I(lower, upper)})
        new_bounds = rule.checked_apply(expr, ctx)
        assert new_bounds[child] == I(None, None)

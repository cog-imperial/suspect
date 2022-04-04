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

import numpy as np
import pyomo.environ as pe
import pytest
from hypothesis import given
import suspect.pyomo.expressions as dex
from suspect.fbbt.tightening.quadratic import *
from suspect.fbbt.tightening.rules import *
from suspect.fbbt.tightening.visitor import BoundsTighteningVisitor
from suspect.interval import Interval as I
from tests.strategies import reals


@pytest.fixture(scope='module')
def visitor():
    return BoundsTighteningVisitor()


@given(reals(), reals())
def test_constraint_rule(visitor, bound1, bound2):
    lower_bound = min(bound1, bound2)
    upper_bound = max(bound1, bound2)
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
        expr_bounds = I(-5, 5)
        children_bounds = [I(1, 2), I(-2, 2), I(-10, 10)]
        new_bounds = self._rule_result(visitor, children_bounds, expr_bounds)
        assert new_bounds[2] == I(-9, 6)


class TestLinearRule:
    children = None
    expr = None

    def _rule_result(self, coefficients, children_bounds, expr_bounds):
        children = [pe.Var() for _ in children_bounds]
        expr = pe.quicksum(
            [child * coef for child, coef in zip(children, coefficients)],
            linear=True
        )

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


class _BilinearTerm:
    def __init__(self, var1, var2, coef):
        self.var1 = var1
        self.var2 = var2
        self.coefficient = coef


class _QuadraticExpression:
    expression_type = ExpressionType.Quadratic

    def __init__(self, children, terms):
        self.args = children
        self.terms = terms


class TestQuadraticRule:
    children = None
    expr = None

    def _rule_result(self, terms, children, children_bounds, expr_bounds):
        expr = _QuadraticExpression(children, terms)

        bounds = ComponentMap()
        bounds[expr] = expr_bounds
        for child, child_bounds in zip(children, children_bounds):
            bounds[child] = child_bounds

        rule = QuadraticRule()
        self.children = children
        self.expr = expr
        return rule.apply(expr, bounds)

    def test_sum_of_squares(self):
        x = pe.Var()
        y = pe.Var()
        new_bounds = self._rule_result(
            [_BilinearTerm(x, x, 1.0), _BilinearTerm(y, y, 2.0)],
            [x, y],
            [I(None, None), I(None, None)],
            I(None, 10),
        )
        assert new_bounds[x] == I(-np.sqrt(10.0), np.sqrt(10))
        assert new_bounds[y] == I(-np.sqrt(5), np.sqrt(5))


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


class _SumExpression:
    expression_type = ExpressionType.Sum

    def __init__(self, children):
        self.children = children

    def nargs(self):
        return len(self.children)


class _LinearExpression:
    expression_type = ExpressionType.Linear

    def __init__(self, coefs, vars):
        self.linear_coefs = coefs
        self.linear_vars = vars
        self.children = vars


class TestUnivariateQuadraticRule:
    def test_result_with_squares(self):
        x = pe.Var()
        y = pe.Var()
        z = pe.Var()
        quadratic = _QuadraticExpression(
            [x, y, z],
            [_BilinearTerm(x, x, 1.0),
             _BilinearTerm(y, y, 1.0),
             _BilinearTerm(z, z, -1.0)]
        )
        linear = _LinearExpression(
            [-5.09144, -19.96611],
            [x, y],
        )
        expr = _SumExpression([linear, quadratic])
        rule = UnivariateQuadraticRule()
        bounds = ComponentMap()
        bounds[x] = I(0, None)
        bounds[y] = I(0, None)
        bounds[z] = I(0, 4.57424778)
        bounds[expr] = I(None, -106.142171)
        result = rule.apply(expr, bounds)
        assert 5.388076 in result[x]
        assert 6.399097 in result[y]
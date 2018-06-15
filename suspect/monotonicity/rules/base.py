# Copyright 2018 Francesco Ceccon
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

"""Monotonicity detection rules for base expressions."""
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.expression import ExpressionType
from suspect.interfaces import Rule


class VariableRule(Rule):
    """Return monotonicity of variable."""
    root_expr = ExpressionType.Variable

    def apply(self, _expr, _ctx):
        return Monotonicity.Nondecreasing


class ConstantRule(Rule):
    """Return monotonicity of constant."""
    root_expr = ExpressionType.Constant

    def apply(self, _expr, _ctx):
        return Monotonicity.Constant


class ConstraintRule(Rule):
    """Return monotonicity of constraint."""
    root_expr = ExpressionType.Constraint

    def apply(self, expr, ctx):
        child = expr.children[0]
        mono = ctx.monotonicity(child)
        if expr.bounded_below() and expr.bounded_above():
            # l <= g(x) <= u
            if mono.is_constant():
                return mono
            return Monotonicity.Unknown
        elif expr.bounded_below():
            # l <= g(x) => -g(x) <= -l
            return mono.negate()
        elif expr.bounded_above():
            # g(x) <= u
            return mono
        raise RuntimeError('Constraint with no bounds')  # pragma: no cover


class ObjectiveRule(Rule):
    """Return monotonicity of objective."""
    root_expr = ExpressionType.Objective

    def apply(self, expr, ctx):
        child = expr.children[0]
        mono = ctx.monotonicity(child)
        if expr.is_minimizing():
            return mono
        # max f(x) == min -f(x)
        return mono.negate()

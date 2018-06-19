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

"""Convexity detection rules for base expressions."""
from suspect.convexity.convexity import Convexity
from suspect.expression import ExpressionType
from suspect.interfaces import Rule


class VariableRule(Rule):
    """Return convexity of variable."""
    root_expr = ExpressionType.Variable

    def apply(self, _expr, _ctx):
        return Convexity.Linear


class ConstantRule(Rule):
    """Return convexity of constant."""
    root_expr = ExpressionType.Constant

    def apply(self, _expr, _ctx):
        return Convexity.Linear


class ConstraintRule(Rule):
    """Return convexity of constraint."""
    root_expr = ExpressionType.Constraint

    def apply(self, expr, ctx):
        child = expr.children[0]
        cvx = ctx.convexity(child)
        if expr.bounded_below() and expr.bounded_above():
            # l <= g(x) <= u
            if cvx.is_linear():
                return cvx
            return Convexity.Unknown
        elif expr.bounded_below():
            # l <= g(x) => -g(x) <= -l
            return cvx.negate()
        elif expr.bounded_above():
            # g(x) <= u
            return cvx
        raise RuntimeError('Constraint with no bounds')  # pragma: no cover


class ObjectiveRule(Rule):
    """Return convexity of objective."""
    root_expr = ExpressionType.Objective

    def apply(self, expr, ctx):
        child = expr.children[0]
        cvx = ctx.convexity(child)
        if expr.is_minimizing():
            return cvx
        # max f(x) == min -f(x)
        return cvx.negate()

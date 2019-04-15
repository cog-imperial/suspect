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
from suspect.monotonicity.rules.rule import MonotonicityRule


class VariableRule(MonotonicityRule):
    """Return monotonicity of variable."""
    def apply(self, _expr, _mono, _bounds):
        return Monotonicity.Nondecreasing


class ConstantRule(MonotonicityRule):
    """Return monotonicity of constant."""
    def apply(self, _expr, _mono, _bounds):
        return Monotonicity.Constant


class ConstraintRule(MonotonicityRule):
    """Return monotonicity of constraint."""
    def apply(self, expr, monotonicity, _bounds):
        assert len(expr.args) == 1
        child = expr.args[0]
        mono = monotonicity[child]
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


class ObjectiveRule(MonotonicityRule):
    """Return monotonicity of objective."""
    def apply(self, expr, monotonicity, bounds):
        assert len(expr.args) == 1
        child = expr.args[0]
        mono = monotonicity[child]
        if expr.is_minimizing():
            return mono
        # max f(x) == min -f(x)
        return mono.negate()

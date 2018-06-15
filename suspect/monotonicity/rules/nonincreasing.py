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

"""Monotonicity detection rules for nonincreasing functions."""
from suspect.expression import UnaryFunctionType, ExpressionType
from suspect.interfaces import UnaryFunctionRule, Rule


class NonincreasingFunctionRule(UnaryFunctionRule):
    """Return monotonicity of nonincreasing function."""
    def apply(self, expr, ctx):
        child = expr.children[0]
        mono = ctx.monotonicity(child)
        return mono.negate()


class AcosRule(NonincreasingFunctionRule):
    """Return monotonicity of acos function."""
    func_type = UnaryFunctionType.Acos


class NegationRule(Rule):
    """Return monotonicity of negation function."""
    root_expr = ExpressionType.Negation

    def apply(self, expr, ctx):
        child = expr.children[0]
        mono = ctx.monotonicity(child)
        return mono.negate()

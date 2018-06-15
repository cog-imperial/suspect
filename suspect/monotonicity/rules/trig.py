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

"""Monotonicity detection rules for trigonometric expressions."""
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.expression import UnaryFunctionType
from suspect.interfaces import UnaryFunctionRule


class SinRule(UnaryFunctionRule):
    """Return monotonicity of sin."""
    func_type = UnaryFunctionType.Sin

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cos_bounds = bounds.cos()
        mono = ctx.monotonicity(child)

        if mono.is_nondecreasing() and cos_bounds.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bounds.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bounds.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nondecreasing() and cos_bounds.is_nonpositive():
            return Monotonicity.Nonincreasing
        return Monotonicity.Unknown # pragma: no cover


class CosRule(UnaryFunctionRule):
    """Return monotonicity of cos."""
    func_type = UnaryFunctionType.Cos

    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        sin_bounds = bounds.sin()
        mono = ctx.monotonicity(child)

        if mono.is_nonincreasing() and sin_bounds.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bounds.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bounds.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nonincreasing() and sin_bounds.is_nonpositive():
            return Monotonicity.Nonincreasing
        return Monotonicity.Unknown

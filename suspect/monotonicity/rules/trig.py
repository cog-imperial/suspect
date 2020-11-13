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
from suspect.monotonicity.rules.rule import MonotonicityRule


class SinRule(MonotonicityRule):
    """Return monotonicity of sin."""
    def apply(self, expr, monotonicity, bounds):
        child = expr.args[0]
        child_bounds = bounds.get(child)
        if child_bounds is None:
            return Monotonicity.Unknown
        cos_bounds = child_bounds.cos()
        mono = monotonicity[child]

        if mono.is_nondecreasing() and cos_bounds.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bounds.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bounds.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nondecreasing() and cos_bounds.is_nonpositive():
            return Monotonicity.Nonincreasing
        return Monotonicity.Unknown # pragma: no cover


class CosRule(MonotonicityRule):
    """Return monotonicity of cos."""
    def apply(self, expr, monotonicity, bounds):
        child = expr.args[0]
        child_bounds = bounds.get(child)
        if child_bounds is None:
            return Monotonicity.Unknown
        sin_bounds = child_bounds.sin()
        mono = monotonicity[child]

        if mono.is_nonincreasing() and sin_bounds.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bounds.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bounds.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nonincreasing() and sin_bounds.is_nonpositive():
            return Monotonicity.Nonincreasing
        return Monotonicity.Unknown

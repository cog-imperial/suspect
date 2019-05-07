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

"""Monotonicity detection rules for linear and sum expressions."""
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.rules.rule import MonotonicityRule


class LinearRule(MonotonicityRule):
    """Return monotonicity of linear expression."""
    def apply(self, expr, mono, bounds):
        monos = [
            _adjust_monotonicity(mono[child], expr.coefficient(child))
            for child in expr.args
        ]
        return _combine_monotonicities(monos)


class SumRule(MonotonicityRule):
    """Return monotonicity of sum expression."""
    def apply(self, expr, monotonicity, bounds):
        monos = [monotonicity[child] for child in expr.args]
        return _combine_monotonicities(monos)


def _adjust_monotonicity(mono, coef):
    if mono.is_unknown() or mono.is_constant():
        return mono

    if coef > 0:
        return mono
    elif coef < 0:
        return mono.negate()

    # if coef == 0, it's a constant with value 0.0
    return Monotonicity.Constant


def _combine_monotonicities(monos):
    all_const = all([m.is_constant() for m in monos])
    all_nondec = all([m.is_nondecreasing() for m in monos])
    all_noninc = all([m.is_nonincreasing() for m in monos])

    if all_const:
        return Monotonicity.Constant
    elif all_nondec:
        return Monotonicity.Nondecreasing
    elif all_noninc:
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown

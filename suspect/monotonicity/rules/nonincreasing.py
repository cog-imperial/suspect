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
from suspect.monotonicity.rules.rule import MonotonicityRule


class NonincreasingFunctionRule(MonotonicityRule):
    """Return monotonicity of nonincreasing function."""
    def apply(self, expr, monotonicity, _bounds):
        child = expr.args[0]
        mono = monotonicity[child]
        return mono.negate()


class AcosRule(NonincreasingFunctionRule):
    """Return monotonicity of acos function."""
    pass


class NegationRule(MonotonicityRule):
    """Return monotonicity of negation function."""
    def apply(self, expr, monotonicity, _bounds):
        child = expr.args[0]
        mono = monotonicity[child]
        return mono.negate()

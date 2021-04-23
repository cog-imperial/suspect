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

"""Monotonicity detection rules for nondecreasing functions."""
from suspect.monotonicity.rules.rule import MonotonicityRule


class NondecreasingFunctionRule(MonotonicityRule):
    """Return monotonicity of nondecreasing function."""
    def apply(self, expr, monotonicity, _bounds):
        child = expr.args[0]
        mono = monotonicity[child]
        return mono


class SqrtRule(NondecreasingFunctionRule):
    """Return monotonicity of sqrt function."""
    pass


class ExpRule(NondecreasingFunctionRule):
    """Return monotonicity of exp function."""
    pass


class LogRule(NondecreasingFunctionRule):
    """Return monotonicity of log function."""
    pass


class Log10Rule(NondecreasingFunctionRule):
    """Return monotonicity of log10 function."""
    pass


class TanRule(NondecreasingFunctionRule):
    """Return monotonicity of tan function."""
    pass


class AsinRule(NondecreasingFunctionRule):
    """Return monotonicity of asin function."""
    pass


class AtanRule(NondecreasingFunctionRule):
    """Return monotonicity of atan function."""
    pass


class ExpressionRule(NondecreasingFunctionRule):
    """Return monotonicity of a named Expression"""
    pass

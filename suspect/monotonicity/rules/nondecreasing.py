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
from suspect.expression import UnaryFunctionType
from suspect.interfaces import UnaryFunctionRule


class NondecreasingFunctionRule(UnaryFunctionRule):
    """Return monotonicity of nondecreasing function."""
    def apply(self, expr, ctx):
        child = expr.children[0]
        mono = ctx.monotonicity(child)
        return mono


class SqrtRule(NondecreasingFunctionRule):
    """Return monotonicity of sqrt function."""
    func_type = UnaryFunctionType.Sqrt


class ExpRule(NondecreasingFunctionRule):
    """Return monotonicity of exp function."""
    func_type = UnaryFunctionType.Exp


class LogRule(NondecreasingFunctionRule):
    """Return monotonicity of log function."""
    func_type = UnaryFunctionType.Log


class TanRule(NondecreasingFunctionRule):
    """Return monotonicity of tan function."""
    func_type = UnaryFunctionType.Tan


class AsinRule(NondecreasingFunctionRule):
    """Return monotonicity of asin function."""
    func_type = UnaryFunctionType.Asin


class AtanRule(NondecreasingFunctionRule):
    """Return monotonicity of atan function."""
    func_type = UnaryFunctionType.Atan

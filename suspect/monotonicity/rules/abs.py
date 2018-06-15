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

"""Monotonicity detection rules for abs function."""
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.expression import UnaryFunctionType
from suspect.interfaces import UnaryFunctionRule


class AbsRule(UnaryFunctionRule):
    """Return monotonicity of abs."""
    func_type = UnaryFunctionType.Abs

    def apply(self, expr, ctx):
        child = expr.children[0]
        mono = ctx.monotonicity(child)
        bounds = ctx.bounds(child)

        if mono.is_constant():
            return mono

        # good examples to understand the behaviour of abs are abs(-x) and
        # abs(1/x)
        if bounds.is_nonnegative():
            # abs(x), x > 0 is the same as x
            return mono
        elif bounds.is_nonpositive():
            # abs(x), x < 0 is the opposite of x
            return mono.negate()
        return Monotonicity.Unknown

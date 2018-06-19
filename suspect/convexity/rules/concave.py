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

"""Convexity detection rules for concave functions."""
from suspect.convexity.convexity import Convexity
from suspect.expression import UnaryFunctionType
from suspect.interfaces import UnaryFunctionRule


class ConcaveFunctionRule(UnaryFunctionRule):
    """Return convexity of concave function."""
    def apply(self, expr, ctx):
        child = expr.children[0]
        bounds = ctx.bounds(child)
        cvx = ctx.convexity(child)
        if bounds.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave
        return Convexity.Unknown

# TODO(fracek): handle sqrt(x*x) which is same as x
class SqrtRule(ConcaveFunctionRule):
    """Return convexity of sqrt."""
    func_type = UnaryFunctionType.Sqrt


class LogRule(ConcaveFunctionRule):
    """Return convexity of log."""
    func_type = UnaryFunctionType.Log

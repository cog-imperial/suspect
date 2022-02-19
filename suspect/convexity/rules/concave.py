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
from suspect.convexity.rules.rule import ConvexityRule


class ConcaveFunctionRule(ConvexityRule):
    """Return convexity of concave function."""
    def apply(self, expr, convexity, _mono, bounds):
        child = expr.args[0]
        bounds = bounds.get(child)
        cvx = convexity[child]
        if bounds is None:
            return Convexity.Unknown
        if bounds.is_nonnegative() and cvx.is_concave():
            return Convexity.Concave
        return Convexity.Unknown


# TODO(fracek): handle sqrt(x*x) which is same as x
class SqrtRule(ConcaveFunctionRule):
    """Return convexity of sqrt."""
    pass


class LogRule(ConcaveFunctionRule):
    """Return convexity of log."""
    pass


class Log10Rule(ConcaveFunctionRule):
    """Return convexity of log10."""
    pass

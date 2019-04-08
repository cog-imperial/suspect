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

"""Convexity detection rules for convex functions."""
from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule


class ConvexFunctionRule(ConvexityRule):
    """Return convexity of convex function."""
    def apply(self, expr, convexity, _mono, _bounds):
        child = expr.args[0]
        cvx = convexity[child]
        if cvx.is_convex():
            return Convexity.Convex
        return Convexity.Unknown


class ExpRule(ConvexFunctionRule):
    """Return convexity of exp."""
    pass

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

"""Convexity detection rules for base expressions."""
from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule


class VariableRule(ConvexityRule):
    """Return convexity of variable."""
    def apply(self, _expr, _convexity, _mono, _bounds):
        return Convexity.Linear


class ConstantRule(ConvexityRule):
    """Return convexity of constant."""
    def apply(self, _expr, _convexity, _mono, _bounds):
        return Convexity.Linear


class ConstraintRule(ConvexityRule):
    """Return convexity of constraint."""
    def apply(self, expr, convexity, _mono, _bounds):
        child = expr.args[0]
        cvx = convexity[child]
        if expr.bounded_below() and expr.bounded_above():
            # l <= g(x) <= u
            if cvx.is_linear():
                return cvx
            return Convexity.Unknown
        elif expr.bounded_below():
            # l <= g(x) => -g(x) <= -l
            return cvx.negate()
        elif expr.bounded_above():
            # g(x) <= u
            return cvx
        raise RuntimeError('Constraint with no bounds')  # pragma: no cover


class ObjectiveRule(ConvexityRule):
    """Return convexity of objective."""
    def apply(self, expr, convexity, _mono, _bounds):
        child = expr.args[0]
        cvx = convexity[child]
        if expr.is_minimizing():
            return cvx
        # max f(x) == min -f(x)
        return cvx.negate()

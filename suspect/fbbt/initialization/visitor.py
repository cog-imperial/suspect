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

"""FBBT bounds initialization visitor."""
from suspect.interval import Interval
from suspect.interfaces import CombineUnaryFunctionRules
from suspect.pyomo.expressions import UnaryFunctionExpression, Constraint
from suspect.visitor import BackwardVisitor
import pyomo.environ as pyo
from suspect.fbbt.initialization.rules import (
    SqrtRule,
    LogRule,
    Log10Rule,
    AsinRule,
    AcosRule,
)

_rule = CombineUnaryFunctionRules({
    'sqrt': SqrtRule(),
    'log': LogRule(),
    'log10': Log10Rule(),
    'asin': AsinRule(),
    'acos': AcosRule()},
    needs_matching_rules=False,
)


def initialize_bounds(expr, bounds):
    if isinstance(expr, pyo.Constraint):
        lower = pyo.value(expr.lower)
        upper = pyo.value(expr.upper)
        return [Interval(lower, upper)]
    if isinstance(expr, UnaryFunctionExpression):
        return _rule.apply(expr, bounds)
    return None


class BoundsInitializationVisitor(BackwardVisitor):
    """Initialize problem bounds using function domains as bound."""
    needs_matching_rules = False

    def visit_expression(self, expr, bounds):
        if isinstance(expr, UnaryFunctionExpression):
            return True, _rule.apply(expr, bounds)
        return False, None

    def handle_result(self, expr, value, bounds):
        if value is None:
            new_bounds = Interval(None, None)
        else:
            new_bounds = value
        bounds[expr] = new_bounds
        return True

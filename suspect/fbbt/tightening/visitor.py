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

"""FBBT bounds tightening visitor."""

from suspect.pyomo.expressions import (
    nonpyomo_leaf_types,
    Constraint,
    Objective,
    ProductExpression,
    ReciprocalExpression,
    LinearExpression,
    SumExpression,
    PowExpression,
    NegationExpression,
    UnaryFunctionExpression,
)
from suspect.visitor import BackwardVisitor
from suspect.interfaces import CombineUnaryFunctionRules
from suspect.interval import Interval
from suspect.expression import ExpressionType as ET
from suspect.fbbt.tightening.rules import (
    ConstraintRule,
    SumRule,
    LinearRule,
    QuadraticRule,
    PowerRule,
    UnaryFunctionRule,
)


_expr_to_rule_map = dict()
_expr_to_rule_map[Constraint] = ConstraintRule()
_expr_to_rule_map[LinearExpression] = LinearRule()
_expr_to_rule_map[SumExpression] = SumRule()
_expr_to_rule_map[PowExpression] = PowerRule()
_expr_to_rule_map[UnaryFunctionExpression] = UnaryFunctionRule()


def tighten_bounds_root_to_leaf(expr, bounds):
    if type(expr) in nonpyomo_leaf_types:
        return None
    if not expr.is_expression_type():
        return None
    rule = _expr_to_rule_map.get(type(expr), None)
    if rule is not None:
        return rule.apply(expr, bounds)
    return None


class BoundsTighteningVisitor(BackwardVisitor):
    """Tighten bounds from sinks to sources."""
    needs_matching_rules = False

    def visit_expression(self, expr, bounds):
        new_bounds = tighten_bounds_root_to_leaf(expr, bounds)
        if new_bounds is not None:
            return True, new_bounds
        return False, None

    def handle_result(self, expr, value, bounds):
        if value is None:
            new_bounds = Interval(None, None)
        else:
            new_bounds = value

        old_bounds = bounds.get(expr, None)
        if old_bounds is not None:
            new_bounds = old_bounds.intersect(new_bounds)
            has_changed = old_bounds != new_bounds
        else:
            has_changed = True
        bounds[expr] = new_bounds
        return has_changed

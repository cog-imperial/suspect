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

"""FBBT bounds propagation visitor."""

from suspect.pyomo.expressions import (
    nonpyomo_leaf_types,
    Var,
    NumericConstant,
    Constraint,
    Objective,
    MonomialTermExpression,
    ProductExpression,
    DivisionExpression,
    ReciprocalExpression,
    LinearExpression,
    SumExpression,
    PowExpression,
    NegationExpression,
    AbsExpression,
    UnaryFunctionExpression,
    _GeneralExpressionData,
    SimpleExpression,
    ScalarExpression,
)
from suspect.visitor import ForwardVisitor
from suspect.pyomo.quadratic import QuadraticExpression
from suspect.fbbt.propagation.rules import (
    VariableRule,
    ConstantRule,
    ConstraintRule,
    ObjectiveRule,
    ProductRule,
    DivisionRule,
    ReciprocalRule,
    LinearRule,
    SumRule,
    PowerRule,
    NegationRule,
    AbsRule,
    QuadraticRule,
    UnaryFunctionRule,
    ExpressionRule,
)


_expr_to_rule_map = dict()
_expr_to_rule_map[NumericConstant] = ConstantRule()
_expr_to_rule_map[Var] = VariableRule()
_expr_to_rule_map[Constraint] = ConstraintRule()
_expr_to_rule_map[Objective] = ObjectiveRule()
_expr_to_rule_map[ProductExpression] = ProductRule()
_expr_to_rule_map[MonomialTermExpression] = ProductRule()
_expr_to_rule_map[DivisionExpression] = DivisionRule()
_expr_to_rule_map[ReciprocalExpression] = ReciprocalRule()
_expr_to_rule_map[LinearExpression] = LinearRule()
_expr_to_rule_map[SumExpression] = SumRule()
_expr_to_rule_map[PowExpression] = PowerRule()
_expr_to_rule_map[NegationExpression] = NegationRule()
_expr_to_rule_map[AbsExpression] = AbsRule()
_expr_to_rule_map[UnaryFunctionExpression] = UnaryFunctionRule()
_expr_to_rule_map[QuadraticExpression] = QuadraticRule()
_expr_to_rule_map[_GeneralExpressionData] = ExpressionRule()
_expr_to_rule_map[SimpleExpression] = ExpressionRule()
_expr_to_rule_map[ScalarExpression] = ExpressionRule()


def propagate_bounds_leaf_to_root(expr, bounds):
    if type(expr) in nonpyomo_leaf_types:
        rule = _expr_to_rule_map[NumericConstant]
    elif expr.is_constant():
        rule = _expr_to_rule_map[NumericConstant]
    elif expr.is_variable_type():
        rule = _expr_to_rule_map[Var]
    else:
        assert expr.is_expression_type()
        rule = _expr_to_rule_map[type(expr)]
    return rule.apply(expr, bounds)


class BoundsPropagationVisitor(ForwardVisitor):
    """Tighten bounds from sources to sinks."""
    intersect_abs_eps = 1e-10

    def visit_expression(self, expr, bounds):
        return True, propagate_bounds_leaf_to_root(expr, bounds)

    def handle_result(self, expr, new_bounds, bounds):
        old_bounds = bounds.get(expr, None)
        if old_bounds is not None:
            # bounds exists, tighten it
            new_bounds = old_bounds.intersect(
                new_bounds,
                abs_eps=self.intersect_abs_eps,
            )
            has_changed = new_bounds != old_bounds
        else:
            has_changed = True

        bounds[expr] = new_bounds
        return has_changed

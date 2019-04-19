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

"""Visitor applying rules for monotonicity propagation."""
from suspect.pyomo.expressions import (
    nonpyomo_leaf_types,
    NumericConstant,
    Var,
    Constraint,
    Objective,
    MonomialTermExpression,
    ProductExpression,
    ReciprocalExpression,
    AbsExpression,
    LinearExpression,
    SumExpression,
    PowExpression,
    NegationExpression,
    UnaryFunctionExpression,
)
from suspect.interfaces import CombineUnaryFunctionRules
from suspect.visitor import Visitor
from suspect.monotonicity.rules import *


_expr_to_rule_map = dict()
_expr_to_rule_map[NumericConstant] = ConstantRule()
_expr_to_rule_map[Var] = VariableRule()
_expr_to_rule_map[Constraint] = ConstraintRule()
_expr_to_rule_map[Objective] = ObjectiveRule()
_expr_to_rule_map[MonomialTermExpression] = ProductRule()
_expr_to_rule_map[ProductExpression] = ProductRule()
_expr_to_rule_map[ReciprocalExpression] = ReciprocalRule()
_expr_to_rule_map[LinearExpression] = LinearRule()
_expr_to_rule_map[SumExpression] = SumRule()
_expr_to_rule_map[PowExpression] = PowerRule()
_expr_to_rule_map[NegationExpression] = NegationRule()
_expr_to_rule_map[AbsExpression] = AbsRule()
_expr_to_rule_map[UnaryFunctionExpression] = CombineUnaryFunctionRules({
    'abs': AbsRule(),
    'sqrt': SqrtRule(),
    'exp': ExpRule(),
    'log': LogRule(),
    'tan': TanRule(),
    'atan': AtanRule(),
    'sin': SinRule(),
    'asin': AsinRule(),
    'cos': CosRule(),
    'acos': AcosRule(),
})


def expression_monotonicity(expr, mono, bounds):
    if type(expr) in nonpyomo_leaf_types:
        rule = _expr_to_rule_map[NumericConstant]
    elif expr.is_constant():
        rule = _expr_to_rule_map[NumericConstant]
    elif expr.is_variable_type():
        rule = _expr_to_rule_map[Var]
    else:
        assert expr.is_expression_type()
        rule = _expr_to_rule_map[type(expr)]
    return rule.apply(expr, mono, bounds)


class MonotonicityPropagationVisitor(Visitor):
    """Visitor applying monotonicity rules."""
    def handle_result(self, expr, result, monotonicity):
        monotonicity[expr] = result
        return True

    def visit_expression(self, expr, mono, bounds):
        return True, expression_monotonicity(expr, mono, bounds)

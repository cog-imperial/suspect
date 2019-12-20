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

"""Visitor applying rules for polynomial degree computation."""
from suspect.polynomial.rules import *  # pylint: disable=wildcard-import
from suspect.pyomo.expressions import (
    NumericConstant,
    Var,
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
)
from suspect.visitor import Visitor

_expr_to_rule_map = dict()
_expr_to_rule_map[NumericConstant] = ConstantRule()
_expr_to_rule_map[Var] = VariableRule()
_expr_to_rule_map[Constraint] = ConstraintRule()
_expr_to_rule_map[Objective] = ObjectiveRule()
_expr_to_rule_map[MonomialTermExpression] = ProductRule()
_expr_to_rule_map[ProductExpression] = ProductRule()
_expr_to_rule_map[DivisionExpression] = DivisionRule()
_expr_to_rule_map[ReciprocalExpression] = ReciprocalRule()
_expr_to_rule_map[LinearExpression] = LinearRule()
_expr_to_rule_map[SumExpression] = SumRule()
_expr_to_rule_map[PowExpression] = PowerRule()
_expr_to_rule_map[NegationExpression] = NegationRule()
_expr_to_rule_map[AbsExpression] = AbsRule()
_expr_to_rule_map[UnaryFunctionExpression] = UnaryFunctionRule()


def expression_polynomial_degree(expr, poly):
    if type(expr) in nonpyomo_leaf_types:
        rule = _expr_to_rule_map[NumericConstant]
    elif expr.is_constant():
        rule = _expr_to_rule_map[NumericConstant]
    elif expr.is_variable_type():
        rule = _expr_to_rule_map[Var]
    else:
        assert expr.is_expression_type()
        rule = _expr_to_rule_map[type(expr)]
    return rule.apply(expr, poly)


class PolynomialDegreeVisitor(Visitor):
    """Visitor applying polynomiality rules."""
    def handle_result(self, expr, result, ctx):
        ctx[expr] = result
        return True

    def visit_expression(self, expr, poly):
        return True, expression_polynomial_degree(expr, poly)

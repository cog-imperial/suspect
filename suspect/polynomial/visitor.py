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
from suspect.visitor import Visitor
from suspect.expression import ExpressionType as ET
from suspect.polynomial.rules import * # pylint: disable=wildcard-import


class PolynomialDegreeVisitor(Visitor):
    """Visitor applying polynomiality rules."""
    def handle_result(self, expr, result, ctx):
        ctx[expr] = result
        return True

    def register_rules(self):
        return {
            ET.Variable: VariableRule(),
            ET.Constant: ConstantRule(),
            ET.Constraint: ConstraintRule(),
            ET.Objective: ObjectiveRule(),
            ET.Division: DivisionRule(),
            ET.Reciprocal: ReciprocalRule(),
            ET.Product: ProductRule(),
            ET.Linear: LinearRule(),
            # QuadraticRule(),
            ET.Sum: SumRule(),
            ET.Negation: NegationRule(),
            ET.Power: PowerRule(),
            ET.UnaryFunction: UnaryFunctionRule(),
        }

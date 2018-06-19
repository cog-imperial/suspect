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
from suspect.interfaces import CombineUnaryFunctionRules
from suspect.visitor import Visitor
from suspect.monotonicity.rules import * # pylint: disable=wildcard-import


class MonotonicityPropagationVisitor(Visitor):
    """Visitor applying monotonicity rules."""
    def handle_result(self, expr, result, ctx):
        ctx.set_monotonicity(expr, result)
        return True

    def register_rules(self):
        return [
            VariableRule(),
            ConstantRule(),
            ConstraintRule(),
            ObjectiveRule(),
            DivisionRule(),
            ProductRule(),
            LinearRule(),
            SumRule(),
            NegationRule(),
            PowerRule(),
            CombineUnaryFunctionRules(
                AbsRule(),
                SqrtRule(),
                ExpRule(),
                LogRule(),
                TanRule(),
                AtanRule(),
                SinRule(),
                AsinRule(),
                CosRule(),
                AcosRule(),
            )
        ]

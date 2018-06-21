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

from suspect.visitor import ForwardVisitor
from suspect.fbbt.propagation.rules import (
    VariableRule,
    ConstantRule,
    ConstraintRule,
    ObjectiveRule,
    ProductRule,
    DivisionRule,
    LinearRule,
    SumRule,
    PowerRule,
    NegationRule,
    UnaryFunctionRule,
)


class BoundsPropagationVisitor(ForwardVisitor):
    """Tighten bounds from sources to sinks."""
    def register_rules(self):
        return [
            VariableRule(),
            ConstantRule(),
            ConstraintRule(),
            ObjectiveRule(),
            ProductRule(),
            DivisionRule(),
            LinearRule(),
            SumRule(),
            PowerRule(),
            NegationRule(),
            UnaryFunctionRule(),
        ]

    def handle_result(self, expr, new_bounds, ctx):
        old_bounds = ctx.get_bounds(expr)
        if old_bounds is not None:
            # bounds exists, tighten it
            new_bounds = old_bounds.intersect(new_bounds)
            has_changed = new_bounds != old_bounds
        else:
            has_changed = True

        ctx.set_bounds(expr, new_bounds)
        # return has_changed
        return True

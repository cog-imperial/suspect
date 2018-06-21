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

from suspect.visitor import BackwardVisitor
from suspect.interfaces import CombineUnaryFunctionRules
from suspect.interval import Interval
from suspect.fbbt.tightening.rules import (
    ConstraintRule,
    SumRule,
    LinearRule,
    PowerRule,
    AbsRule,
    SqrtRule,
    ExpRule,
    LogRule,
)


class BoundsTighteningVisitor(BackwardVisitor):
    """Tighten bounds from sinks to sources."""
    needs_matching_rules = False

    def register_rules(self):
        return [
            ConstraintRule(),
            SumRule(),
            LinearRule(),
            PowerRule(),
            CombineUnaryFunctionRules(
                AbsRule(),
                SqrtRule(),
                ExpRule(),
                LogRule(),
                needs_matching_rules=False,
            )
        ]

    def handle_result(self, expr, value, ctx):
        if value is None:
            new_bounds = Interval(None, None)
        else:
            new_bounds = value

        old_bounds = ctx.get_bounds(expr)
        if old_bounds is not None:
            new_bounds = old_bounds.intersect(new_bounds)
            has_changed = old_bounds != new_bounds
        else:
            has_changed = True
        ctx.set_bounds(expr, new_bounds)
        # return has_changed
        return True

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
from suspect.visitor import BackwardVisitor
from suspect.fbbt.initialization.rules import (
    SqrtRule,
    LogRule,
    AsinRule,
    AcosRule,
)


class BoundsInitializationVisitor(BackwardVisitor):
    """Initialize problem bounds using function domains as bound."""
    needs_matching_rules = False

    def register_rules(self):
        return [
            CombineUnaryFunctionRules(
                SqrtRule(),
                LogRule(),
                AsinRule(),
                AcosRule(),
            )
        ]

    def handle_result(self, expr, value, ctx):
        if value is None:
            new_bounds = Interval(None, None)
        else:
            new_bounds = value
        ctx.set_bounds(expr, new_bounds)
        return True

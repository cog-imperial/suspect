# Copyright 2017 Francesco Ceccon
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

"""FBBT bounds initialization."""
import suspect.dag.expressions as dex
from suspect.dag.visitor import BackwardVisitor
from suspect.interval import Interval


class BoundsInitializationVisitor(BackwardVisitor):
    """Initialize problem bounds using function domains as bound."""
    def register_callbacks(self):
        return {
            dex.SqrtExpression: self._visit_sqrt,
            dex.LogExpression: self._visit_log,
            dex.AsinExpression: self._visit_asin,
            dex.AcosExpression: self._visit_acos,
        }

    def handle_result(self, expr, value, ctx):
        if value is None:
            new_bound = Interval(None, None)
            has_changed = True
        else:
            new_bound = value
            has_changed = True
        ctx[expr] = new_bound
        return has_changed

    def _visit_sqrt(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Interval(0, None),
        }

    def _visit_log(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Interval(0, None)
        }

    def _visit_asin(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Interval(-1, 1)
        }

    def _visit_acos(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Interval(-1, 1)
        }

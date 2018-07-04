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

# from suspect.monotonicity.monotonicity import Monotonicity
# from suspect.convexity.convexity import Convexity

"""Module with classes for extending SUSPECT."""
from suspect.visitor import ForwardVisitor


class ConvexityDetector(ForwardVisitor):
    """Convexity Detector base class."""
    needs_matching_rules = False

    def register_rules(self):
        raise NotImplementedError('register_rules')

    def handle_result(self, expr, result, ctx):
        if result is None:
            return False
        ctx.convexity[expr] = result
        return not result.is_unknown()

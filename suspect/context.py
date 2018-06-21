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

# This is an hack while we transition to the new visitors. Remove ASAP.
# We want to be able to access values in ctx both as ctx.mono[exp] and
# as ctx.mono(expr)
class _HackDict:
    def __init__(self):
        self._d = {}

    def get(self, key, default=None):
        return self._d.get(key, default)

    def __call__(self, key):
        return self._d[key]

    def __getitem__(self, key):
        return self._d[key]

    def __contains__(self, key):
        return key in self._d

    def __setitem__(self, key, value):
        self._d[key] = value


class SpecialStructurePropagationContext(object):
    def __init__(self):
        self.bounds = _HackDict()
        self.bound = self.bounds # TODO(fracek): for now, remove this later
        self.monotonicity = _HackDict()
        self.convexity = _HackDict()
        self.polynomial = _HackDict()

    def set_bounds(self, expr, value):
        self.bounds[expr] = value

    def get_bounds(self, expr):
        return self.bounds.get(expr)

    def set_monotonicity(self, expr, value):
        self.monotonicity[expr] = value

    def set_convexity(self, expr, value):
        self.convexity[expr] = value

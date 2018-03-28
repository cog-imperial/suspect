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

from enum import Enum


class Convexity(Enum):
    Convex = 0
    Concave = 1
    Linear = 2
    Unknown = 3

    def is_convex(self):
        return self == self.Convex or self == self.Linear

    def is_concave(self):
        return self == self.Concave or self == self.Linear

    def is_linear(self):
        return self == self.Linear

    def is_unknown(self):
        return self == self.Unknown

    def negate(self):
        if self.is_linear():
            return self.Linear
        elif self.is_convex():
            return self.Concave
        elif self.is_concave():
            return self.Convex
        else:
            return self.Unknown

    def combine(self, other):
        """Combines two Convexity objects together.

        If both objects are the same type (linear, convex, concave) then
        the resulting object will be of the same type, otherwise it will
        be Convexity.Unknown
        """
        if self.is_linear() and other.is_linear():
            return self.Linear
        elif self.is_convex() and other.is_convex():
            return self.Convex
        elif self.is_concave() and other.is_concave():
            return self.Concave
        else:
            return self.Unknown

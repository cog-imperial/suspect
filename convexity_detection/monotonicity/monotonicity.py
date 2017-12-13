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


class Monotonicity(Enum):
    Nondecreasing = 0
    Nonincreasing = 1
    Constant = 2
    Unknown = 3

    def is_nondecreasing(self):
        return self == self.Nondecreasing or self == self.Constant

    def is_nonincreasing(self):
        return self == self.Nonincreasing or self == self.Constant

    def is_constant(self):
        return self == self.Constant

    def is_unknown(self):
        return self == self.Unknown
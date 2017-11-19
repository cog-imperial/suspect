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

from convexity_detection.monotonicity.monotonicity import Monotonicity
from collections import defaultdict


def linear_monotonicity(handler, expr):
    def _adjust_monotonicity(mono, coef):
        if mono.is_unknown() or mono.is_constant():
            return mono

        if coef > 0:
            return mono
        elif coef < 0:
            if mono.is_nondecreasing():
                return Monotonicity.Nonincreasing
            else:
                return Monotonicity.Nondecreasing
        else:
            # if coef == 0, it's a constant with value 0.0
            return Monotonicity.Constant

    if hasattr(expr, '_coef'):
        coefs = expr._coef
    else:
        coefs = defaultdict(lambda: 1.0)

    monos = [
        _adjust_monotonicity(handler.monotonicity(a), coefs[id(a)])
        for a in expr._args
    ]
    all_nondec = all([m.is_nondecreasing() for m in monos])
    all_noninc = all([m.is_nonincreasing() for m in monos])

    if all_nondec:
        return Monotonicity.Nondecreasing
    elif all_noninc:
        return Monotonicity.Nonincreasing
    else:
        return Monotonicity.Unknown

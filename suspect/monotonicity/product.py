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

from suspect.monotonicity.monotonicity import Monotonicity


def product_monotonicity(mono_handler, expr):
    assert len(expr.children) == 2

    def _product_mono(f, g):
        mono_f = mono_handler.get(f)
        mono_g = mono_handler.get(g)

        if mono_f.is_constant() and mono_g.is_constant():
            return Monotonicity.Constant
        elif mono_g.is_constant():
            if mono_f.is_nondecreasing() and mono_handler.is_nonnegative(g):
                return Monotonicity.Nondecreasing
            elif mono_f.is_nonincreasing() and mono_handler.is_nonpositive(g):
                return Monotonicity.Nondecreasing
            elif mono_f.is_nondecreasing() and mono_handler.is_nonpositive(g):
                return Monotonicity.Nonincreasing
            elif mono_f.is_nonincreasing() and mono_handler.is_nonnegative(g):
                return Monotonicity.Nonincreasing
            else:
                return Monotonicity.Unknown
        elif mono_f.is_constant():
            return _product_mono(g, f)
        else:
            nondec_cond1 = (
                mono_f.is_nondecreasing() and mono_handler.is_nonnegative(g)
            ) or (
                mono_f.is_nonincreasing() and mono_handler.is_nonpositive(g)
            )
            nondec_cond2 = (
                mono_handler.is_nonnegative(f) and mono_g.is_nondecreasing()
            ) or (
                mono_handler.is_nonpositive(f) and mono_g.is_nonincreasing()
            )
            noninc_cond1 = (
                mono_f.is_nonincreasing() and mono_handler.is_nonnegative(g)
            ) or (
                mono_f.is_nondecreasing() and mono_handler.is_nonpositive(g)
            )
            noninc_cond2 = (
                mono_handler.is_nonnegative(f) and mono_g.is_nonincreasing()
            ) or (
                mono_handler.is_nonpositive(f) and mono_g.is_nondecreasing()
            )

            if nondec_cond1 and nondec_cond2:
                return Monotonicity.Nondecreasing
            elif noninc_cond1 and noninc_cond2:
                return Monotonicity.Nonincreasing
            else:
                return Monotonicity.Unknown

    f, g = expr.children
    return _product_mono(f, g)

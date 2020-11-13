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

"""Monotonicity detection rules for product expressions."""
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.rules.rule import MonotonicityRule


class ProductRule(MonotonicityRule):
    """Return monotonicity of product."""
    def apply(self, expr, monotonicity, bounds):
        f, g = expr.args
        return _product_monotonicity(f, g, bounds, monotonicity)


def _product_monotonicity(f, g, bounds, monotonicity):
    mono_f = monotonicity[f]
    mono_g = monotonicity[g]
    bound_f = bounds.get(f)
    bound_g = bounds.get(g)

    if mono_f.is_constant() and mono_g.is_constant():
        return Monotonicity.Constant
    elif mono_g.is_constant():
        return _product_with_constant_g(f, g, mono_f, mono_g, bound_f, bound_g)
    elif mono_f.is_constant():
        return _product_with_constant_g(g, f, mono_g, mono_f, bound_g, bound_f)

    if bound_g is None or bound_f is None:
        return Monotonicity.Unknown
    # no constant function involved
    nondec_cond1 = (
        mono_f.is_nondecreasing() and bound_g.is_nonnegative()
    ) or (
        mono_f.is_nonincreasing() and bound_g.is_nonpositive()
    )
    nondec_cond2 = (
        bound_f.is_nonnegative() and mono_g.is_nondecreasing()
    ) or (
        bound_f.is_nonpositive() and mono_g.is_nonincreasing()
    )
    noninc_cond1 = (
        mono_f.is_nonincreasing() and bound_g.is_nonnegative()
    ) or (
        mono_f.is_nondecreasing() and bound_g.is_nonpositive()
    )
    noninc_cond2 = (
        bound_f.is_nonnegative() and mono_g.is_nonincreasing()
    ) or (
        bound_f.is_nonpositive() and mono_g.is_nondecreasing()
    )

    if nondec_cond1 and nondec_cond2:
        return Monotonicity.Nondecreasing
    elif noninc_cond1 and noninc_cond2:
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown  # pragma: no cover


def _product_with_constant_g(f, g, mono_f, _mono_g, _bound_f, bound_g):
    if bound_g is None:
        return Monotonicity.Unknown
    if mono_f.is_nondecreasing() and bound_g.is_nonnegative():
        return Monotonicity.Nondecreasing
    elif mono_f.is_nonincreasing() and bound_g.is_nonpositive():
        return Monotonicity.Nondecreasing
    elif mono_f.is_nondecreasing() and bound_g.is_nonpositive():
        return Monotonicity.Nonincreasing
    elif mono_f.is_nonincreasing() and bound_g.is_nonnegative():
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown  # pragma: no cover

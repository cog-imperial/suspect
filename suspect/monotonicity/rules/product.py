
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
from suspect.expression import ExpressionType
from suspect.interfaces import Rule


class ProductRule(Rule):
    """Return monotonicity of product."""
    root_expr = ExpressionType.Product

    def apply(self, expr, ctx):
        f, g = expr.children
        return _product_monotonicity(f, g, ctx)


def _product_monotonicity(f, g, ctx):
    mono_f = ctx.monotonicity(f)
    mono_g = ctx.monotonicity(g)
    bound_f = ctx.bounds(f)
    bound_g = ctx.bounds(g)
    if mono_f.is_constant() and mono_g.is_constant():
        return Monotonicity.Constant
    elif mono_g.is_constant():
        return _product_with_constant_g(f, g, mono_f, mono_g, bound_f, bound_g)
    elif mono_f.is_constant():
        return _product_with_constant_g(g, f, mono_g, mono_f, bound_g, bound_f)
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
    return Monotonicity.Unknown # pragma: no cover


def _product_with_constant_g(f, g, mono_f, _mono_g, _bound_f, bound_g):
    if mono_f.is_nondecreasing() and bound_g.is_nonnegative():
        return Monotonicity.Nondecreasing
    elif mono_f.is_nonincreasing() and bound_g.is_nonpositive():
        return Monotonicity.Nondecreasing
    elif mono_f.is_nondecreasing() and bound_g.is_nonpositive():
        return Monotonicity.Nonincreasing
    elif mono_f.is_nonincreasing() and bound_g.is_nonnegative():
        return Monotonicity.Nonincreasing
    return Monotonicity.Unknown # pragma: no cover

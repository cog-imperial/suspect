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

from suspect.convexity.convexity import Convexity


def division_convexity(expr, ctx):
    assert len(expr.children) == 2
    f, g = expr.children

    mono_f = ctx.monotonicity[f]
    mono_g = ctx.monotonicity[g]
    bound_f = ctx.bound[f]
    bound_g = ctx.bound[g]

    if mono_g.is_constant():
        cvx_f = ctx.convexity[f]

        if cvx_f.is_convex() and bound_g.is_positive():
            return Convexity.Convex
        elif cvx_f.is_concave() and bound_g.is_negative():
            return Convexity.Convex
        elif cvx_f.is_concave() and bound_g.is_positive():
            return Convexity.Concave
        elif cvx_f.is_convex() and bound_g.is_negative():
            return Convexity.Concave

    elif mono_f.is_constant():
        cvx_g = ctx.convexity[g]

        # want to avoid g == 0
        if 0 in bound_g:
            return Convexity.Unknown

        if bound_f.is_nonnegative():
            if bound_g.is_positive() and cvx_g.is_concave():
                return Convexity.Convex
            elif bound_g.is_negative() and cvx_g.is_convex():
                return Convexity.Concave
            else:
                return Convexity.Unknown
        elif bound_f.is_nonpositive():
            if bound_g.is_negative() and cvx_g.is_convex():
                return Convexity.Convex
            elif bound_g.is_positive() and cvx_g.is_concave():
                return Convexity.Concave
            else:
                return Convexity.Unknown
        else:
            return Convexity.Unknown

    return Convexity.Unknown

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

from suspect.bounds import expression_bounds
from suspect.convexity.convexity import Convexity
from suspect.math import (
    cos, pi, almosteq, almostlte, almostgte, mpf
)
import pyomo.environ as aml


def cos_convexity(bound, cvx, arg):
    # if bound is like [0, pi], we need to be extra carefull
    diff = bound.u - bound.l
    if diff > pi:
        return Convexity.Unknown

    cos_l = cos(bound.l)
    cos_u = cos(bound.u)
    if cos_l * cos_u < 0 and not almosteq(cos_l*cos_u, 0):
        return Convexity.Unknown

    l = bound.l % (2 * pi)
    u = l + diff

    sin_bound = expression_bounds(aml.sin(arg))

    if almostlte(u, mpf('1/2')*pi) or almostgte(l, mpf('3/2')*pi):
        cond_linear = cvx.is_linear()
        cond_convex = (
            cvx.is_convex() and sin_bound.is_nonpositive()
        )
        cond_concave = (
            cvx.is_concave() and sin_bound.is_nonnegative()
        )

        if cond_linear or cond_convex or cond_concave:
            return Convexity.Concave
    else:
        cond_linear = cvx.is_linear()
        cond_concave = (
            cvx.is_concave() and sin_bound.is_nonpositive()
        )
        cond_convex = (
            cvx.is_convex() and sin_bound.is_nonnegative()
        )

        if cond_linear or cond_concave or cond_convex:
            return Convexity.Convex

    return Convexity.Uknown


def acos_convexity(bound, cvx):
    bound_in_neg_1_0 = (
        almostlte(-1, bound.l) and
        almostlte(bound.l, bound.u) and
        almostlte(bound.u, 0)
    )
    if bound_in_neg_1_0 and cvx.is_concave():
        return Convexity.Convex

    bound_in_0_1 = (
        almostlte(0, bound.l) and
        almostlte(bound.l, bound.u) and
        almostlte(bound.u, 1)
    )
    if bound_in_0_1 and cvx.is_convex():
        return Convexity.Concave

    return Convexity.Unknown
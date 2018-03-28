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

from suspect.math.arbitrary_precision import pi
from suspect.convexity.convexity import Convexity


def tan_convexity(expr, ctx):
    arg = expr.children[0]
    bound = ctx.bound[arg]
    cvx = ctx.convexity[arg]

    if 2.0*bound.size() > pi:
        return Convexity.Unknown

    tan_bound = bound.tan()
    if tan_bound.lower_bound * tan_bound.upper_bound < 0:
        return Convexity.Unknown

    if tan_bound.is_nonnegative() and cvx.is_convex():
        return Convexity.Convex

    if tan_bound.is_nonpositive() and cvx.is_concave():
        return Convexity.Concave

    return Convexity.Unknown


def atan_convexity(expr, ctx):
    arg = expr.children[0]
    bound = ctx.bound[arg]
    cvx = ctx.convexity[arg]

    if bound.is_nonpositive() and cvx.is_convex():
        return Convexity.Convex

    if bound.is_nonnegative() and cvx.is_concave():
        return Convexity.Concave

    return Convexity.Unknown

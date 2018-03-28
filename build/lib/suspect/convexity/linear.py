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


def linear_convexity(expr, ctx):
    def _adjust_convexity(cvx, coef):
        if cvx.is_unknown() or cvx.is_linear():
            return cvx

        if coef > 0:
            return cvx
        elif coef < 0:
            return cvx.negate()
        else:
            # if coef == 0, it's a constant with value 0.0
            return Convexity.Linear

    if hasattr(expr, 'coefficients'):
        coefs = expr.coefficients
    else:
        coefs = [1.0] * len(expr.children)

    cvxs = [
        _adjust_convexity(ctx.convexity[a], coef)
        for a, coef in zip(expr.children, coefs)
    ]
    all_linear = all([c.is_linear() for c in cvxs])
    all_convex = all([c.is_convex() for c in cvxs])
    all_concave = all([c.is_concave() for c in cvxs])

    if all_linear:
        return Convexity.Linear
    elif all_convex:
        return Convexity.Convex
    elif all_concave:
        return Convexity.Concave

    return Convexity.Unknown

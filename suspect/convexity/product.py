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
import suspect.dag.expressions as dex


def _product_convexity(handler, f, g):
    cvx_f = handler.convexity(f)
    if cvx_f.is_linear():
        return cvx_f
    elif cvx_f.is_convex() and handler.is_nonnegative(g):
        return Convexity.Convex
    elif cvx_f.is_concave() and handler.is_nonpositive(g):
        return Convexity.Convex
    elif cvx_f.is_concave() and handler.is_nonnegative(g):
        return Convexity.Concave
    elif cvx_f.is_convex() and handler.is_nonpositive(g):
        return Convexity.Concave
    else:
        return Convexity.Unknown


def product_convexity(handler, expr):
    assert(len(expr.children) == 2)
    f = expr.children[0]
    g = expr.children[1]
    mono_f = handler.monotonicity(f)
    mono_g = handler.monotonicity(g)

    if f is g:
        # Same f**2
        cvx_f = handler.convexity(f)
        if cvx_f.is_linear():
            return Convexity.Convex
        elif cvx_f.is_convex() and handler.is_nonnegative(f):
            return Convexity.Convex
        elif cvx_f.is_concave() and handler.is_nonpositive(f):
            return Convexity.Convex

    # -c*x*x is encoded as Linear([-c], [x])*Variable(x)
    if isinstance(f, dex.LinearExpression) and isinstance(g, dex.Variable):
        if len(f.children) == 1 and f.children[0] is g:
            if f.coefficients[0] > 0:
                return Convexity.Convex
            else:
                return Convexity.Concave

    if isinstance(g, dex.LinearExpression) and isinstance(f, dex.Variable):
        if len(g.children) == 1 and g.children[0] is f:
            if f.coefficients[0] > 0:
                return Convexity.Convex
            else:
                return Convexity.Concave

    if mono_g.is_constant():
        return _product_convexity(handler, f, g)
    elif mono_f.is_constant():
        return _product_convexity(handler, g, f)

    return Convexity.Unknown

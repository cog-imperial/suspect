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

from suspect.util import numeric_types, numeric_value
from suspect.convexity.convexity import Convexity
from suspect.expr_visitor import LinearExpression, Variable


def _product_convexity(handler, f, g):
    assert(isinstance(g, numeric_types))
    g = numeric_value(g)
    cvx_f = handler.convexity(f)
    if cvx_f.is_convex() and handler.is_nonnegative(g):
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
    assert(len(expr._args) == 2)
    f = expr._args[0]
    g = expr._args[1]
    if isinstance(g, numeric_types):
        return _product_convexity(handler, f, g)
    elif isinstance(f, numeric_types):
        return _product_convexity(handler, g, f)

    # Try to detected convexity of expression like x*x or
    # (x + a)*(x + b), where a and b are constants
    if isinstance(f, LinearExpression) and len(f._args) == 1:
        if isinstance(f._args[0], Variable):
            f = f._args[0]

    if isinstance(g, LinearExpression) and len(g._args) == 1:
        if isinstance(g._args[0], Variable):
            g = g._args[0]

    if isinstance(f, Variable) and isinstance(g, Variable):
        if f is g:
            # x^2
            return Convexity.Convex

    return Convexity.Unknown

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


def division_convexity(handler, expr):
    assert(len(expr._args) == 2)
    f, g = expr._args

    if isinstance(g, numeric_types):
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

    elif isinstance(f, numeric_types):
        f = numeric_value(f)
        cvx_g = handler.convexity(g)

        # want to avoid g == 0
        if not handler.is_positive(g) or not handler.is_negative(g):
            return Convexity.Unknown

        elif handler.is_nonnegative(g) and cvx_g.is_concave():
            return Convexity.Convex
        elif handler.is_nonpositive(g) and cvx_g.is_convex():
            return Convexity.Convex
        elif handler.is_nonnegative(g) and cvx_g.is_convex():
            return Convexity.Concave
        elif handler.is_nonpositive(g) and cvx_g.is_concave():
            return Convexity.Concave

    return Convexity.Unknown

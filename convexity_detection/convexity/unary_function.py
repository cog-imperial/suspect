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

from convexity_detection.convexity.convexity import Convexity
from convexity_detection.convexity.sin import (
    sin_convexity,
    asin_convexity,
)
from convexity_detection.convexity.cos import (
    cos_convexity,
    acos_convexity,
)
from convexity_detection.convexity.tan import (
    tan_convexity,
    atan_convexity,
)


def unary_function_convexity(handler, expr):
    assert len(expr._args) == 1
    name = expr._name
    arg = expr._args[0]
    bound = handler.bound(arg)
    cvx = handler.convexity(arg)

    if name == 'sqrt':
        # TODO: handle sqrt(x*x) which is same as x
        if handler.is_nonnegative(arg) and cvx.is_concave():
            return Convexity.Concave

    elif name == 'exp':
        if cvx.is_convex():
            return Convexity.Convex

    elif name == 'log':
        # TODO: handle log(exp(x)) == x
        if handler.is_positive(arg) and cvx.is_concave():
            return Convexity.Concave

    elif name == 'sin':
        return sin_convexity(bound, cvx, arg)

    elif name == 'cos':
        return cos_convexity(bound, cvx, arg)

    elif name == 'tan':
        return tan_convexity(bound, cvx)

    elif name == 'asin':
        return asin_convexity(bound, cvx)

    elif name == 'acos':
        return acos_convexity(bound, cvx)

    elif name == 'atan':
        return atan_convexity(bound, cvx)

    else:
        raise RuntimeError('unknown unary function {}'.format(name))

    return Convexity.Unknown

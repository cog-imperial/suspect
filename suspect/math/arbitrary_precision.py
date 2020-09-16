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

# pylint: disable=invalid-name
"""Arbitrary precision mathematical constants and comparison."""
import mpmath


mpf = mpmath.mpf
make_number = mpmath.mpf
inf = make_number('inf')
zero = make_number(0)
pi = mpmath.pi
isnan = mpmath.isnan
isinf = mpmath.isinf


def _declare_unary_function(name, fun):
    # skip rounding mode since we don't round
    globals()[name] = lambda n, _: fun(n)


def _declare_binary_function(name, fun):
    # skip rounding mode since we don't round
    globals()[name] = lambda x, y, _: fun(x, y)


_UNARY_FUNCTIONS = ['sqrt', 'log', 'log10', 'exp', 'sin', 'asin', 'cos', 'acos', 'tan', 'atan']
for fun in _UNARY_FUNCTIONS:
    _declare_unary_function(fun, getattr(mpmath, fun))

_BINARY_FUNCTIONS = ['power']
for fun in _BINARY_FUNCTIONS:
    _declare_binary_function(fun, getattr(mpmath, fun))


def down(f):
    """Perform computation rounding down."""
    return f()


def up(f):
    """Perform computation rounding down."""
    return f()


def min_(*args):
    """Returns the minimum."""
    return min(a for a in args if not isnan(a))


def max_(*args):
    """Return the maximum."""
    return max(a for a in args if not isnan(a))


def almosteq(a, b, rel_eps=None, abs_eps=None):
    """Floating point equality check between `a` and `b`."""
    # in mpmath inf != inf, but we want inf == inf
    if abs(a) == inf and abs(b) == inf:
        return (a > 0 and b > 0) or (a < 0 and b < 0)
    return mpmath.almosteq(a, b, rel_eps=rel_eps, abs_eps=abs_eps)


def almostgte(a, b):
    """Return True if a >= b."""
    return a > b or almosteq(a, b)


def almostlte(a, b):
    """Return True if a <= b."""
    return a < b or almosteq(a, b)

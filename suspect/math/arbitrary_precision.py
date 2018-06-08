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
from typing import Any, Callable
import mpmath


mpf = mpmath.mpf
make_number = mpmath.mpf
inf = make_number('inf')
zero = make_number(0)
pi = mpmath.pi
isnan = mpmath.isnan


def _declare_function(name, fun):
    globals()[name] = lambda n, _: fun(n)


_FUNCTIONS = ['sqrt', 'log', 'exp', 'sin', 'asin', 'cos', 'acos', 'tan', 'atan']
for fun in _FUNCTIONS:
    _declare_function(fun, getattr(mpmath, fun))


def down(f: Callable[[Any], Any]) -> Any:
    """Perform computation rounding down."""
    return f()


def up(f: Callable[[Any], Any]) -> Any:
    """Perform computation rounding down."""
    return f()


def min_(*args):
    """Returns the minimum."""
    return min(a for a in args if not isnan(a))


def max_(*args):
    """Return the maximum."""
    return max(a for a in args if not isnan(a))


def almosteq(a: Any, b: Any) -> bool:
    """Floating point equality check between `a` and `b`."""
    # in mpmath inf != inf, but we want inf == inf
    if abs(a) == inf and abs(b) == inf:
        return (a > 0 and b > 0) or (a < 0 and b < 0)
    return mpmath.almosteq(a, b)


def almostgte(a: Any, b: Any) -> bool:
    """Return True if a >= b."""
    return a > b or almosteq(a, b)


def almostlte(a: Any, b: Any) -> bool:
    """Return True if a <= b."""
    return a < b or almosteq(a, b)

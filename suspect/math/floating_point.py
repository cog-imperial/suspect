# Copyright 2019 Francesco Ceccon
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
"""Floating point arithmetic."""
import numpy as np

make_number = np.float
inf = np.inf
zero = float(0.0)
pi = np.pi
isnan = lambda n: np.isnan(n)
isinf = lambda n: np.isinf(n)


def _declare_unary_function(name, fun):
    # skip rounding mode since we don't round
    globals()[name] = lambda n, _: float(fun(n))


def _declare_binary_function(name, fun):
    # skip rounding mode since we don't round
    globals()[name] = lambda x, y, _: float(fun(x, y))


_UNARY_FUNCTIONS = [
    ('sqrt', 'sqrt'), ('log', 'log'), ('log10', 'log10'), ('exp', 'exp'), ('sin', 'sin'),
    ('asin', 'arcsin'), ('cos', 'cos'), ('acos', 'arccos'),
    ('tan', 'tan'), ('atan', 'arctan')
]
for fun_name, fun in _UNARY_FUNCTIONS:
    _declare_unary_function(fun_name, getattr(np, fun))


_BINARY_FUNCTIONS = ['power']
for fun in _BINARY_FUNCTIONS:
    _declare_binary_function(fun, getattr(np, fun))


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
    # inf != inf, but we want inf == inf
    if rel_eps is None:
        rel_eps = 1.0e-5
    if abs_eps is None:
        abs_eps = 1.0e-8
    return np.all(np.isclose(a, b, rtol=rel_eps, atol=abs_eps))


def almostgte(a, b):
    """Return True if a >= b."""
    return np.all(np.logical_or(a > b, almosteq(a, b)))


def almostlte(a, b):
    """Return True if a <= b."""
    return np.all(np.logical_or(a < b, almosteq(a, b)))

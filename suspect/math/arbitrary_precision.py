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

import sys
import mpmath


mpf = mpmath.mpf
inf = mpf('inf')

# Re-export symbols from mpmath
_EXPORTED_SYMBOLS = [
    'pi', 'sin', 'cos', 'sqrt', 'log', 'sin', 'asin', 'cos', 'acos',
    'tan', 'atan', 'exp']

_module = sys.modules[__name__]
for sym in _EXPORTED_SYMBOLS:
    setattr(_module, sym, getattr(mpmath, sym))


isnan = mpmath.isnan


def almosteq(a, b):
    """Floating point equality check between `a` and `b`."""
    if abs(a) == inf and abs(b) == inf:
        return True
    return mpmath.almosteq(a, b)


def almostgte(a, b):
    """Return True if a >= b."""
    return a > b or almosteq(a, b)


def almostlte(a, b):
    """Return True if a <= b."""
    return a < b or almosteq(a, b)

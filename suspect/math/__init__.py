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

"""SUSPECT math mode.


Selecting Math Mode
-------------------

To set the math mode used by SUSPECT you need to call the ``set_math_mode`` function:


.. code-block:: python

   from suspect.math import set_math_mode, MathMode

   set_math_mode(MathMode.ARBITRARY_PRECISION)


Adding a new Math Mode
----------------------

A math mode needs to define two constants:

 * ``inf``: infinity value
 * ``pi``: the trig constant
 * ``zero``: 0.0

It also needs to define the following functions:

 * ``make_number``: return a new number
 * ``isnan``: predicate to test if it's Not-a-Number
 * ``almosteq``: predicate to test if two numbers are equal
 * ``almostgte```: predicate to test if two numbers are >=
 * ```almostlte``: predicate to test if two numbers are <=
 * ``down``: perform computation rounding down
 * ``up``: perform computation rounding up

The following functions take two parameters, a number and a RoundMode:

 * ``sqrt``
 * ``log``
 * ``log10``
 * ``exp``
 * ``sin``
 * ``cos``
 * ``tan``
 * ``asin``
 * ``acos``
 * ``atan``
"""

# TODO(fracek): make math mode open like the remaining of suspect.
# TODO(fracek): Then add plugin for correctly rounded arithmetic.


class MathMode:
    """Math mode used internally by SUSPECT."""
    ARBITRARY_PRECISION = 1
    FLOATING_POINT = 2


class RoundMode:
    """Round mode to use for computation.

     * RN: round to nearest
     * RU: round towards +inf
     * RD: round towards -inf
     * RZ: round to zero
    """
    RN = 1
    RU = 2
    RD = 3
    RZ = 4


# pylint: disable=undefined-all-variable
_COMMON_MEMBERS = [
    'make_number',
    'inf',
    'pi',
    'zero',
    'sin',
    'cos',
    'sqrt',
    'log',
    'log10',
    'exp',
    'power',
    'sin',
    'asin',
    'cos',
    'acos',
    'tan',
    'atan',
    'isnan',
    'isinf',
    'almosteq',
    'almostgte',
    'almostlte',
    'down',
    'up',
    'min_',
    'max_',
]

# TODO(fracek): drop this, for now it's need for compat with old code
_ARBITRARY_PRECISION_MEMBERS = ['mpf']

__all__ = _COMMON_MEMBERS


def set_math_mode(math_mode):
    """Set the math mode used by SUSPECT.

    Parameters
    ----------
    math_mode: MathMode
        the math mode to use
    """
    if math_mode == MathMode.ARBITRARY_PRECISION:
        from suspect.math import arbitrary_precision as arb
        for member in _COMMON_MEMBERS:
            globals()[member] = getattr(arb, member)
        for member in _ARBITRARY_PRECISION_MEMBERS:
            globals()[member] = getattr(arb, member)
    elif math_mode == MathMode.FLOATING_POINT:
        from suspect.math import floating_point as fp
        for member in _COMMON_MEMBERS:
            globals()[member] = getattr(fp, member)
    else:
        raise RuntimeError('Invalid MathMode')


set_math_mode(MathMode.FLOATING_POINT)

# Copyright 2018 Francesco Ceccon
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

"""Generic expression support."""
from enum import IntEnum


class ExpressionType(IntEnum):
    """Expression types supported by SUSPECT."""
    Variable = 1
    Constant = 2
    Constraint = 3
    Objective = 4
    Division = 5
    Product = 6
    Linear = 7
    Sum = 8
    Power = 9
    # While negation is technically an unary function, it's equivalent to a linear
    # expression with coefficient -1
    Negation = 10
    UnaryFunction = 11
    Quadratic = 12
    Reciprocal = 13


class UnaryFunctionType(IntEnum):
    """Unary function type supported by SUSPECT."""
    Abs = 1
    Sqrt = 2
    Exp = 3
    Log = 4
    Sin = 5
    Cos = 6
    Tan = 7
    Asin = 8
    Acos = 9
    Atan = 10

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

"""Monotonicity rules."""
from .base import VariableRule, ConstantRule, ConstraintRule, ObjectiveRule
from .abs import AbsRule
from .division import DivisionRule, ReciprocalRule
from .linear import LinearRule, SumRule
from .nondecreasing import SqrtRule, ExpRule, LogRule, Log10Rule, TanRule, AsinRule, AtanRule, ExpressionRule
from .nonincreasing import AcosRule, NegationRule
from .pow import PowerRule
from .product import ProductRule
from .quadratic import QuadraticRule
from .trig import SinRule, CosRule


__all__ = [
    'VariableRule', 'ConstantRule', 'ConstraintRule', 'ObjectiveRule',
    'ProductRule', 'DivisionRule', 'LinearRule', 'SumRule', 'AbsRule',
    'SqrtRule', 'ExpRule', 'LogRule', 'TanRule', 'AsinRule', 'AtanRule',
    'AcosRule', 'NegationRule', 'PowerRule', 'SinRule', 'CosRule',
    'QuadraticRule', 'ReciprocalRule', 'Log10Rule', 'ExpressionRule'
]

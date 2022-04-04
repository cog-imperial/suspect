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

# pylint: skip-file
import pytest
import numpy as np
import hypothesis
from collections import namedtuple
from suspect.interval import Interval
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.convexity.convexity import Convexity
from suspect.expression import ExpressionType as ET
from suspect.context import SpecialStructurePropagationContext


hypothesis.settings.register_profile('github-ci', max_examples=10)

BilinearTerm = namedtuple('BilinearTerm', ['var1', 'var2', 'coefficient'])


class PlaceholderExpression(object):
    depth = 0
    def __init__(self, expression_type=None, children=None, coefficients=None,
                 is_constant=False, value=None, constant_term=None, terms=None,
                 bounded_above=False, bounded_below=False,
                 lower_bound=None, upper_bound=None,
                 is_minimizing=False, func_type=None):
        self.expression_type = expression_type
        self.children = children
        if coefficients is not None:
            self._coefficients = dict([(ch, co) for ch, co in zip(children, coefficients)])
        else:
            self._coefficients = {}

        self.is_constant = lambda: is_constant
        self.constant_term = constant_term
        self.terms = terms
        self.value = value
        self.bounded_above = lambda: bounded_above
        self.bounded_below = lambda: bounded_below
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_minimizing = lambda: is_minimizing
        self.func_type = func_type

    def coefficient(self, expr):
        return self._coefficients[expr]

    def is_variable_type(self):
        return self.expression_type == ET.Variable

    def is_expression_type(self):
        return not (self.is_variable_type() or self.is_constant())

    @property
    def args(self):
        return self.children

    @property
    def lb(self):
        return self.lower_bound

    @property
    def ub(self):
        return self.upper_bound

@pytest.fixture
def ctx():
    return SpecialStructurePropagationContext()


def bound_description_to_bound(bound_str):
    if isinstance(bound_str, str):
        return {
            'zero': Interval.zero(),
            'nonpositive': Interval(None, 0),
            'nonnegative': Interval(0, None),
            'positive': Interval(1, None),
            'negative': Interval(None, -1),
            'unbounded': Interval(None, None),
        }[bound_str]
    elif isinstance(bound_str, Interval):
        return bound_str
    else:
        return Interval(bound_str, bound_str)


def mono_description_to_mono(mono_str):
    return {
        'nondecreasing': Monotonicity.Nondecreasing,
        'nonincreasing': Monotonicity.Nonincreasing,
        'constant': Monotonicity.Constant,
        'unknown': Monotonicity.Unknown,
    }[mono_str]


def cvx_description_to_cvx(cvx_str):
    return {
        'convex': Convexity.Convex,
        'concave': Convexity.Concave,
        'linear': Convexity.Linear,
        'unknown': Convexity.Unknown,
    }[cvx_str]

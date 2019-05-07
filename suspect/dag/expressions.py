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

from enum import Enum
import abc
import sys
from suspect.math import inf # pylint: disable=no-name-in-module


class Domain(Enum):
    """The variable domain"""
    REALS = 0
    INTEGERS = 1
    BINARY = 2


class Sense(Enum):
    """The objective function sense"""
    MINIMIZE = 0
    MAXIMIZE = 1


class Expression(metaclass=abc.ABCMeta):
    """The base class for all expressions objects in the DAG"""
    is_source = False
    is_sink = False

    def __init__(self, children=None):
        if children is None:
            children = []

        self._children = children

    @property
    def args(self):
        return self._children

    @property
    def args(self):
        return self._children

    def is_variable_type(self):
        return False

    def is_constant(self):
        return False


class Objective(Expression):
    is_sink = True

    def __init__(self, name, sense=None, children=None):
        super().__init__(children)
        if sense is None:
            sense = Sense.MINIMIZE
        self.sense = sense
        self.name = name

    @property
    def body(self):
        return self.args[0]

    def is_expression_type(self):
        return True

    def is_minimizing(self):
        return self.sense == Sense.MINIMIZE

    def is_maximizing(self):
        return self.sense == Sense.MAXIMIZE


class BoundedExpression(Expression):
    def __init__(self, lower_bound, upper_bound, children=None):
        super().__init__(children)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def bounded_below(self):
        return self.lower_bound is not None and self.lower_bound != -inf

    def bounded_above(self):
        return self.upper_bound is not None and self.upper_bound != inf


class Constraint(BoundedExpression):
    is_sink = True

    def __init__(self, name, lower_bound, upper_bound, children=None):
        super().__init__(lower_bound, upper_bound, children)
        self.name = name

    @property
    def body(self):
        return self.args[0]

    def is_expression_type(self):
        return True

    def is_equality(self):
        return self.lower_bound == self.upper_bound

    def __str__(self):
        return 'Constraint(name={}, lower_bound={}, upper_bound={}, args={})'.format(
            self.name, self.lower_bound, self.upper_bound, self.args
        )


class Variable(BoundedExpression):
    is_source = True

    def __init__(self, name, lower_bound, upper_bound, domain=None):
        super().__init__(lower_bound, upper_bound, None)
        self.domain = domain
        self.name = name

    def is_binary(self):
        return self.domain == Domain.BINARY

    def is_integer(self):
        return self.domain == Domain.INTEGERS

    def is_real(self):
        return self.domain == Domain.REALS

    def is_constant(self):
        # TODO(fracek): if we ever support fixing variables, change this
        return False

    def __str__(self):
        return 'Variable(name={}, lower_bound={}, upper_bound={}, domain={})'.format(
            self.name, self.lower_bound, self.upper_bound, self.domain
        )


class Constant(BoundedExpression):
    is_source = True

    def __init__(self, value):
        super().__init__(value, value, None)

    @property
    def value(self):
        assert self.lower_bound == self.upper_bound
        return self.lower_bound

    def is_constant(self):
        return True

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
from suspect.math.arbitrary_precision import inf


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

        self._depth = 0
        self._update_depth()

    @property
    def depth(self):
        """The depth of the expression.

        The depth of the expression is defined as `0` if the
        expression is a source (Variables and Constants), otherwise
        it is the maximum depth of its children plus `1`.
        """
        return self._depth

    @property
    def children(self):
        return self._children

    def _update_depth(self):
        max_depth = self.depth
        for child in self.children:
            if child.depth >= max_depth:
                max_depth = child.depth + 1
        self._depth = max_depth


class ProductExpression(Expression):
    pass


class DivisionExpression(Expression):
    pass


class SumExpression(Expression):
    pass


class PowExpression(Expression):
    pass


class LinearExpression(Expression):
    def __init__(self, coefficients=None, children=None, constant_term=None):
        super().__init__(children)
        if coefficients is None:
            coefficients = []
        if constant_term is None:
            constant_term = 0.0
        self.constant_term = constant_term
        self.coefficients = coefficients
        self._check_coefficients()

    def _check_coefficients(self):
        assert len(self.coefficients) == len(self.children)


class UnaryFunctionExpression(Expression):
    def __init__(self, children=None):
        super().__init__(children)
        assert len(self.children) == 1


class NegationExpression(UnaryFunctionExpression):
    func_name = 'negation'


class AbsExpression(UnaryFunctionExpression):
    func_name = 'abs'


class SqrtExpression(UnaryFunctionExpression):
    func_name = 'sqrt'


class ExpExpression(UnaryFunctionExpression):
    func_name = 'exp'


class LogExpression(UnaryFunctionExpression):
    func_name = 'log'


class SinExpression(UnaryFunctionExpression):
    func_name = 'sin'


class CosExpression(UnaryFunctionExpression):
    func_name = 'cos'


class TanExpression(UnaryFunctionExpression):
    func_name = 'tan'


class AsinExpression(UnaryFunctionExpression):
    func_name = 'asin'


class AcosExpression(UnaryFunctionExpression):
    func_name = 'acos'


class AtanExpression(UnaryFunctionExpression):
    func_name = 'atan'


class Objective(Expression):
    is_sink = True

    def __init__(self, name, sense=None, children=None):
        super().__init__(children)
        if sense is None:
            sense = Sense.MINIMIZE
        self.sense = sense
        self.name = name


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

    def linear_component(self):
        pass

    def nonlinear_component(self):
        pass

    def __str__(self):
        return 'Constraint(name={}, lower_bound={}, upper_bound={}, children={})'.format(
            self.name, self.lower_bound, self.upper_bound, self.children
        )


class Variable(BoundedExpression):
    is_source = True

    def __init__(self, name, lower_bound, upper_bound, domain=None):
        super().__init__(lower_bound, upper_bound, None)
        self.domain = domain
        self.name = name

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
        return self.lower_bound

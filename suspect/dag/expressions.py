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
from suspect.expression import ExpressionType, UnaryFunctionType
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
        self._parents = []

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

    @property
    def parents(self):
        return self._parents

    def add_parent(self, parent):
        self._parents.append(parent)

    def _update_depth(self):
        max_depth = self.depth
        for child in self.children:
            if child.depth >= max_depth:
                max_depth = child.depth + 1
        self._depth = max_depth

    def is_constant(self):
        return False

    def __hash__(self):
        return hash(id(self))

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self == other)


class ProductExpression(Expression):
    expression_type = ExpressionType.Product


class DivisionExpression(Expression):
    expression_type = ExpressionType.Division


class SumExpression(Expression):
    expression_type = ExpressionType.Sum


class PowExpression(Expression):
    expression_type = ExpressionType.Power


class LinearExpression(Expression):
    expression_type = ExpressionType.Linear

    def __init__(self, coefficients=None, children=None, constant_term=None):
        super().__init__(children)
        if coefficients is None:
            coefficients = []
        if constant_term is None:
            constant_term = 0.0
        self.constant_term = constant_term
        self._coefficients = dict([(child, coef) for child, coef in zip(children, coefficients)])
        self._check_coefficients()

    def coefficient(self, expr):
        return self._coefficients[expr]

    def _check_coefficients(self):
        assert len(self._coefficients) == len(self.children)


class UnaryFunctionExpression(Expression):
    expression_type = ExpressionType.UnaryFunction
    def __init__(self, children=None):
        super().__init__(children)
        assert len(self.children) == 1


class NegationExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.Negation
    func_name = 'negation'


class AbsExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Abs
    func_name = 'abs'


class SqrtExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Sqrt
    func_name = 'sqrt'


class ExpExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Exp
    func_name = 'exp'


class LogExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Log
    func_name = 'log'


class SinExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Sin
    func_name = 'sin'


class CosExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Cos
    func_name = 'cos'


class TanExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Tan
    func_name = 'tan'


class AsinExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Asin
    func_name = 'asin'


class AcosExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Acos
    func_name = 'acos'


class AtanExpression(UnaryFunctionExpression):
    expression_type = ExpressionType.UnaryFunction
    func_type = UnaryFunctionType.Atan
    func_name = 'atan'


class Objective(Expression):
    expression_type = ExpressionType.Objective
    is_sink = True

    def __init__(self, name, sense=None, children=None):
        super().__init__(children)
        if sense is None:
            sense = Sense.MINIMIZE
        self.sense = sense
        self.name = name

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
    expression_type = ExpressionType.Constraint
    is_sink = True

    def __init__(self, name, lower_bound, upper_bound, children=None):
        super().__init__(lower_bound, upper_bound, children)
        self.name = name

    def linear_component(self):
        linear, _ = self._split_linear_nonlinear()
        return linear

    def nonlinear_component(self):
        _, nonlinear = self._split_linear_nonlinear()
        return nonlinear

    def _split_linear_nonlinear(self):
        child = self.children[0]

        if isinstance(child, LinearExpression):
            return child, None

        if not isinstance(child, SumExpression):
            return None, [child]

        linear = None
        nonlinear = []
        for arg in child.children:
            if isinstance(arg, LinearExpression):
                if linear is not None:
                    raise AssertionError(
                        'Constraint root should have only one LinearExpression child'
                    )
                linear = arg
            else:
                nonlinear.append(arg)
        return linear, nonlinear

    def is_equality(self):
        return self.lower_bound == self.upper_bound

    def __str__(self):
        return 'Constraint(name={}, lower_bound={}, upper_bound={}, children={})'.format(
            self.name, self.lower_bound, self.upper_bound, self.children
        )


class Variable(BoundedExpression):
    expression_type = ExpressionType.Variable
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
    expression_type = ExpressionType.Constant
    is_source = True

    def __init__(self, value):
        super().__init__(value, value, None)

    @property
    def value(self):
        assert self.lower_bound == self.upper_bound
        return self.lower_bound

    def is_constant(self):
        return True

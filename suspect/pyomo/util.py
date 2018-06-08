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

from numbers import Number
import pyomo.environ as aml
from suspect.pyomo.expr_visitor import (
    InequalityExpression,
    EqualityExpression,
)
from suspect.interval import Interval

numeric_types = (int, Number, aml.NumericConstant)


def numeric_value(n):
    if isinstance(n, (int, Number)):
        return n
    elif isinstance(n, aml.NumericConstant):
        return n.value
    else:
        raise ValueError('must be one of numeric_types')


def bounds_and_expr(expr):
    if isinstance(expr, InequalityExpression):
        return _inequality_bounds_and_expr(expr)
    elif isinstance(expr, EqualityExpression):
        return _equality_bounds_and_expr(expr)
    else:
        raise ValueError('expr must be InequalityExpression or EqualityExpression')


def _inequality_bounds_and_expr(expr):
    if len(expr._args) == 2:
        (lhs, rhs) = expr._args
        if isinstance(lhs, aml.NumericConstant):
            return Interval(lhs.value, None), rhs
        else:
            return Interval(None, rhs.value), lhs
    elif len(expr._args) == 3:
        (lhs, ex, rhs) = expr._args
        return Interval(lhs.value, rhs.value), ex
    else:
        raise ValueError('Malformed InequalityExpression')


def _equality_bounds_and_expr(expr):
    if len(expr._args) == 2:
        body, rhs = expr._args
        return Interval(rhs.value, rhs.value), body
    else:
        raise ValueError('Malformed EqualityExpression')


def model_variables(model):
    """Return a list of variables in the model"""
    for variables in model.component_map(aml.Var, active=True).itervalues():
        for idx in variables:
            yield variables[idx]


def model_constraints(model):
    """Return a list of constraints in the model"""
    for cons in model.component_map(aml.Constraint, active=True).itervalues():
        for idx in cons:
            yield cons[idx]


def model_objectives(model):
    """Return a list of objectives in the model"""
    for obj in model.component_map(aml.Objective, active=True).itervalues():
        for idx in obj:
            yield obj[idx]

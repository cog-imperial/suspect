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
from pyomo.core.expr.numeric_expr import nonpyomo_leaf_types
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.expr.logical_expr import (
    InequalityExpression,
    EqualityExpression,
    RangedExpression,
)
from suspect.interval import Interval


def is_numeric(expr):
    return (
        expr.__class__ in nonpyomo_leaf_types or
        isinstance(expr, NumericConstant)
    )


def numeric_value(expr):
    if expr.__class__ in nonpyomo_leaf_types:
        return float(expr)
    assert isinstance(expr, NumericConstant)
    return float(expr.value)


def bounds_and_expr(expr):
    if isinstance(expr, (InequalityExpression, RangedExpression)):
        return _inequality_bounds_and_expr(expr)
    elif isinstance(expr, EqualityExpression):
        return _equality_bounds_and_expr(expr)
    else:
        raise ValueError('expr must be InequalityExpression or EqualityExpression')


def _inequality_bounds_and_expr(expr):
    if len(expr._args_) == 2:
        (lhs, rhs) = expr._args_
        if is_numeric(lhs):
            return Interval(numeric_value(lhs), None), rhs
        else:
            return Interval(None, numeric_value(rhs)), lhs
    elif len(expr._args_) == 3:
        (lhs, ex, rhs) = expr._args_
        return Interval(numeric_value(lhs), numeric_value(rhs)), ex
    else:
        raise ValueError('Malformed InequalityExpression')


def _equality_bounds_and_expr(expr):
    if len(expr._args_) == 2:
        body, rhs = expr._args_
        return Interval(numeric_value(rhs), numeric_value(rhs)), body
    else:
        raise ValueError('Malformed EqualityExpression')


def model_variables(model):
    """Return a list of variables in the model"""
    for v in model.component_data_objects(aml.Var, active=True, descend_into=True, sort=True):
        yield v


def model_constraints(model):
    """Return a list of constraints in the model"""
    for c in model.component_data_objects(aml.Constraint, active=True, descend_into=True, sort=True):
        yield c


def model_objectives(model):
    """Return a list of objectives in the model"""
    for obj in model.component_data_objects(aml.Objective, active=True, descend_into=True, sort=True):
        yield obj

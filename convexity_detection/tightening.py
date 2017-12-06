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

from convexity_detection.bounds import Bound, expression_bounds
from convexity_detection.expr_visitor import (
    ExpressionHandler,
    bottom_up_visit as visit_expression,
    EqualityExpression,
    InequalityExpression,
    LinearExpression,
    SumExpression,
    ProductExpression,
)
from convexity_detection.expr_dict import TightestExpressionDict
from convexity_detection.util import (
    model_variables,
    model_constraints,
    model_objectives,
    bounds_and_expr,
)
from convexity_detection.math import *


def tighten_model_bounds(model):
    """Find the tightest bounds for the model variables.

    Parameters
    ----------
    model: Model
        the Pyomo model

    Returns
    -------
    bounds: ExpressionDict
        the dictionary with the variables bounds
    """
    bounds = TightestExpressionDict()
    for obj in model_objectives(model):
        tighten_bounds_from_context(obj, bounds)
    reverse_bound_tightening(model, bounds)
    return bounds


def inequality_bounds(expr):
    """Returns the bounds of the constraint as `Bound`.

    Given constraint c^L <= g(x) <= c^U, returns Bound(c^L, c^U).
    """
    (bounds, _) = bounds_and_expr(expr)
    return bounds


def _join_expressions(exprs):
    """Join a list of expressions in a single expression.

    A list of LinearExpression is a LinearExpression, while a
    list of other expressions is joined as sum.
    """
    if len(exprs) > 0:
        return sum(exprs)
    else:
        return None


def linear_nonlinear_components(expr):
    """Returns the linear and nonlinear components of the constraint.

    Given a constraint g(x) = L(x) + N(x), returns (L(x), N(x))
    """
    assert isinstance(expr, (InequalityExpression, EqualityExpression))

    (_, expr) = bounds_and_expr(expr)

    if isinstance(expr, SumExpression):
        linear = _join_expressions([
            e for e in expr._args
            if isinstance(e, LinearExpression)
        ])
        nonlinear = _join_expressions([
            e for e in expr._args
            if not isinstance(e, LinearExpression)
        ])
        return linear, nonlinear
    elif isinstance(expr, ProductExpression) and len(expr._args) == 1:
        return expr, None
    elif isinstance(expr, LinearExpression):
        return expr, None
    else:
        return None, expr


def _remove_var_from_linear_expr(expr, x):
    assert isinstance(expr, LinearExpression)
    args = [a for a in expr._args if a is not x]
    new_linear = sum(expr._coef[id(a)] * a for a in args)
    coef = expr._coef.get(id(x))
    return new_linear, coef


def reverse_bound_tightening(model, bounds, max_iter=10):
    """Perform reverse bound tightening on the model.

    This function will tighten the bounds of the model variables using
    the information on the constraints.

    Reverse bound tightening is an iterative process, a maximum number
    of `max_iter` steps will be performed.
    """

    for numiter in range(max_iter):
        bounds_changed = reverse_bound_tightening_step(model, bounds)

        if not bounds_changed:
            break

    return numiter


def reverse_bound_tightening_step(model, bounds):
    """Perform a step of reverse bound tightening.

    For each variable in the problem, will find the tightest bounds
    on variables.

    Returns True if the at least one bound has changed.
    """

    has_any_changed = False
    for constraint in model_constraints(model):
        has_changed = tighten_bounds_from_context(constraint, bounds)
        if has_changed:
            has_any_changed = True
        for variable in model_variables(model):
            var_bounds = tighten_variable_bounds(constraint, variable, bounds)
            old_var_bounds = bounds[variable]
            if var_bounds != old_var_bounds:
                has_any_changed = True
                bounds.tighten(variable, var_bounds)

    return has_any_changed


def tighten_variable_bounds(constraint, x, bounds):
    """Tightens the bounds of `x` based on the bounds of constraint.

    Given constraint g(x) = l(x) + a*x + N(x), where l(x) = L(x) - a*x,
    compute the new bounds on x as:

        c^L - l^U - n^U <= a*x <= c^U - l^L - n^L

    where:

        l^L <= l(x) <= l^U
        n^L <= N(x) <= n^U
    """
    expr = constraint.expr

    # TODO: what to do if it's an eq constraint?

    var_bounds = bounds[x] or Bound(x.bounds[0], x.bounds[1])

    (linear, nonlinear) = linear_nonlinear_components(expr)

    if linear is not None:
        linear_no_x, x_coef = _remove_var_from_linear_expr(linear, x)
        if x_coef is None:
            # var not found
            return var_bounds
        linear_bounds = expression_bounds(linear_no_x, bounds)
    else:
        # handle constraints of the type l <= x <= u
        bounds, sub_expr = bounds_and_expr(expr)
        if sub_expr is x:
            return bounds
        # Can't update bounds without linear part
        return var_bounds

    starting_bounds = inequality_bounds(expr)

    if nonlinear is not None:
        nonlinear_bounds = expression_bounds(nonlinear, bounds)
    else:
        nonlinear_bounds = Bound(0, 0)

    candidate_lower = (
        starting_bounds.l - linear_bounds.u - nonlinear_bounds.u
        )/x_coef

    candidate_upper = (
        starting_bounds.u - linear_bounds.l - nonlinear_bounds.l
        )/x_coef

    # flip inequalities if x_coef < 0
    if x_coef < 0:
        candidate_lower, candidate_upper = candidate_upper, candidate_lower

    new_lower = max(candidate_lower, var_bounds.l)
    new_upper = min(candidate_upper, var_bounds.u)

    return Bound(new_lower, new_upper)


class BoundsFromContextHandler(ExpressionHandler):
    def __init__(self, memo):
        self.has_changed = False
        self.memo = memo

    def bound(self, expr):
        return self.memo[expr]

    def set_bound(self, expr, bound):
        self.memo.tighten(expr, bound)

    def visit_number(self, n):
        pass

    def visit_numeric_constant(self, expr):
        pass

    def visit_variable(self, expr):
        pass

    def visit_equality(self, expr):
        assert(len(expr._args) == 2)
        bounds, body = bounds_and_expr(expr)
        old_bounds = self.bound(body)
        self.set_bound(body, bounds)
        if bounds != old_bounds:
            self.has_changed = True

    def visit_inequality(self, expr):
        bounds, body = bounds_and_expr(expr)
        old_bounds = self.bound(body)
        self.set_bound(body, bounds)
        if bounds != old_bounds:
            self.has_changed = True

    def visit_product(self, expr):
        pass

    def visit_division(self, expr):
        pass

    def visit_sum(self, expr):
        pass

    def visit_linear(self, expr):
        pass

    def visit_negation(self, expr):
        pass

    def visit_unary_function(self, expr):
        # TODO: should be a top-down visitor so that we can
        # also set the bound based on parent bounds.
        # e.g. 0 <= sqrt(x) <= 2  -->  0 <= x <= 4
        assert len(expr._args) == 1

        name = expr._name
        arg = expr._args[0]
        old_bound = self.bound(arg)
        if name == 'sqrt':
            self.set_bound(arg, Bound(0, None))
            if old_bound != Bound(0, None):
                self.has_changed = True
        elif name == 'log':
            self.set_bound(arg, Bound(0, None))
            if old_bound != Bound(0, None):
                self.has_changed = True
        elif name == 'asin':
            self.set_bound(arg, Bound(-1, 1))
            if old_bound != Bound(-1, 1):
                self.has_changed = True
        elif name == 'acos':
            self.set_bound(arg, Bound(-1, 1))
            if old_bound != Bound(-1, 1):
                self.has_changed = True

    def visit_abs(self, expr):
        pass

    def visit_pow(self, expr):
        pass


def tighten_bounds_from_context(constraint, bounds):
    handler = BoundsFromContextHandler(bounds)
    visit_expression(handler, constraint.expr)
    return handler.has_changed

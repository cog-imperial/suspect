from convexity_detection.bounds import Bound, expr_bounds
from convexity_detection.expr_visitor import (
    EqualityExpression,
    InequalityExpression,
    LinearExpression,
    SumExpression,
    ProductExpression,
)
from convexity_detection.util import (
    model_variables,
    model_constraints,
)
import pyomo.environ as aml
import copy as pycopy


def _bounds_and_expr(expr):
    if len(expr._args) == 2:
        (lhs, rhs) = expr._args
        if isinstance(lhs, aml.NumericConstant):
            return Bound(lhs.value, None), rhs
        else:
            return Bound(None, rhs.value), lhs
    elif len(expr._args) == 3:
        (lhs, ex, rhs) = expr._args
        return Bound(lhs.value, rhs.value), ex
    else:
        raise ValueError('Malformed InequalityExpression')


def inequality_bounds(expr):
    """Returns the bounds of the constraint as `Bound`.

    Given constraint c^L <= g(x) <= c^U, returns Bound(c^L, c^U).
    """
    (bounds, _) = _bounds_and_expr(expr)
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

    (_, expr) = _bounds_and_expr(expr)

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


def remove_var_from_linear_expr(expr, x):
    assert isinstance(expr, LinearExpression)
    args = [a for a in expr._args if a is not x]
    new_linear = sum(expr._coef[id(a)] * a for a in args)
    coef = expr._coef.get(id(x))
    return new_linear, coef


def reverse_bound_tightening_step(model):
    """Perform a step of reverse bound tightening.

    For each variable in the problem, will find the tightest bounds
    on variables.

    Returns a map of id(variable) => (variable, bounds)
    """

    new_variables_bounds = {}
    for constraint in model_constraints(model):
        for variable in model_variables(model):
            bounds = tighten_variable_bounds(constraint, variable)
            if id(variable) in new_variables_bounds:
                _, previous_bounds = new_variables_bounds[id(variable)]
                new_bounds = Bound(
                    max(bounds.l, previous_bounds.l),
                    min(bounds.u, previous_bounds.u),
                )
                new_variables_bounds[id(variable)] = (variable, new_bounds)
            else:
                new_variables_bounds[id(variable)] = (variable, bounds)

    return new_variables_bounds


def reverse_bound_tightening(model, copy=True, max_iter=10):
    """Perform reverse bound tightening on the model.

    This function will tighten the bounds of the model variables using
    the information on the constraints.

    If `copy` is `True`, will work and return a copy of the original
    model.

    Reverse bound tightening is an iterative process, a maximum number
    of `max_iter` steps will be performed.
    """
    if copy:
        model = pycopy.deepcopy(model)

    prev_variables_bounds = None
    for numiter in range(max_iter):
        new_variables_bounds = reverse_bound_tightening_step(model)

        for variable, new_bounds in new_variables_bounds.values():
            variable.setlb(new_bounds.l)
            variable.setub(new_bounds.u)

        # Check if the bounds didn't change in this step, meaning
        # we converged.
        if prev_variables_bounds is not None:
            bounds_changed = False
            for variable, new_bounds in new_variables_bounds.values():
                _, old_bounds = prev_variables_bounds.get(id(variable))
                if old_bounds is not None and old_bounds != new_bounds:
                    bounds_changed = True
        else:
            bounds_changed = True

        if not bounds_changed:
            break

        prev_variables_bounds = new_variables_bounds

    return (model, numiter)


def tighten_variable_bounds(constraint, x):
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

    var_bounds = Bound(x.bounds[0], x.bounds[1])

    (linear, nonlinear) = linear_nonlinear_components(expr)

    if linear is not None:
        linear_no_x, x_coef = remove_var_from_linear_expr(linear, x)
        if x_coef is None:
            # var not found
            return var_bounds
        linear_bounds = expr_bounds(linear)
    else:
        # Can't update bounds without linear part
        return var_bounds

    starting_bounds = inequality_bounds(expr)

    if nonlinear is not None:
        nonlinear_bounds = expr_bounds(nonlinear)
    else:
        nonlinear_bounds = Bound(0, 0)

    new_lower = max(
        (starting_bounds.l - linear_bounds.u - nonlinear_bounds.u)/x_coef,
        var_bounds.l,
    )
    new_upper = min(
        (starting_bounds.u - linear_bounds.l - nonlinear_bounds.l)/x_coef,
        var_bounds.u,
    )
    return Bound(new_lower, new_upper)

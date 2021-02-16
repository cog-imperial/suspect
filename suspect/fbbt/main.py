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

"""Class to run bound tightener or a problem."""
import pyomo.environ as pe

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.common.collections import ComponentMap

from suspect.interval import Interval
from suspect.fbbt.initialization import initialize_bounds
from suspect.fbbt.propagation import propagate_bounds_leaf_to_root
from suspect.fbbt.tightening import tighten_bounds_root_to_leaf


def perform_fbbt(model, max_iter=10, active=True, objective_bounds=None, initial_bounds=None, should_continue=None):
    """Perform FBBT on the model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        the pyomo model
    max_iter : int
        maximum number of FBBT iterations
    active : boolean
        perform FBBT only on active constraints
    objective_bounds : dict-like
        map between an objective and its lower and upper bound
    initial_bounds : dict-like
        map with existing bounds
    should_continue : callable
        predicate that returns False if fbbt should stop early

    Returns
    -------
    ComponentMap
        tightened bounds for variables and expressions
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    if initial_bounds is None:
        bounds = ComponentMap()
    else:
        bounds = initial_bounds

    if should_continue is None:
        should_continue = lambda: True

    if objective_bounds is None:
        objective_bounds = ComponentMap()

    objectives = list(model.component_data_objects(pe.Objective, active=active, descend_into=True))
    constraints = list(model.component_data_objects(pe.Constraint, active=active, descend_into=True))

    for constraint in constraints:
        lower = pe.value(constraint.lower)
        upper = pe.value(constraint.upper)
        bounds[constraint.body] = Interval(lower, upper)

    for objective in objectives:
        obj_lower, obj_upper = objective_bounds.get(objective, (None, None))
        bounds[objective.expr] = Interval(obj_lower, obj_upper)

    for var in model.component_data_objects(pe.Var, active=active, descend_into=True):
        lower = pe.value(var.lb)
        upper = pe.value(var.ub)
        bounds[var] = Interval(lower, upper)

    if not should_continue():
        return bounds

    _perform_fbbt_step(model, bounds, -1, active, should_continue, constraints, objectives)

    for iter in range(max_iter):
        changed = _perform_fbbt_step(model, bounds, iter, active, should_continue, constraints, objectives)
        if not changed:
            break
        if not should_continue():
            break

    return bounds


def _perform_fbbt_step(model, bounds, iter, active, should_continue, constraints, objectives):
    if iter < 0:
        func = _initialize_then_propagate_bounds_on_expr
    else:
        func = _tighten_then_propagate_bounds_on_expr

    changed = False
    for objective in objectives:
        changed |= func(objective.expr, bounds)
        if not should_continue():
            return changed

    for constraint in constraints:
        changed |= func(constraint.body, bounds)
        if not should_continue():
            return changed

    return changed


def _initialize_then_propagate_bounds_on_expr(expr, bounds):
    return _perform_fbbt_iteration_on_expr(
        expr, bounds, initialize_bounds, propagate_bounds_leaf_to_root, visit_all=True
    )


def _tighten_then_propagate_bounds_on_expr(expr, bounds):
    return _perform_fbbt_iteration_on_expr(
        expr, bounds, tighten_bounds_root_to_leaf, propagate_bounds_leaf_to_root
    )


def _perform_fbbt_iteration_on_expr(expr, bounds, tighten, propagate, visit_all=False):
    def enter_node(node):
        result = tighten(node, bounds)
        if result is None:
            return None, None
        if isinstance(result, list):
            result_iter = zip(node.args, result)
        else:
            result_iter = result.items()
        descend_into_args = []
        for arg, value in result_iter:
            changed = _update_bounds_map(bounds, arg, value)
            if changed:
                descend_into_args.append(arg)

        if visit_all:
            return None, None

        return descend_into_args, None

    def exit_node(node, data):
        result = propagate(node, bounds)
        return _update_bounds_map(bounds, node, result)

    return StreamBasedExpressionVisitor(
        enterNode=enter_node,
        exitNode=exit_node,
    ).walk_expression(expr)


def _update_bounds_map(bounds, expr, value):
    if value is None:
        new_bounds = Interval(None, None)
    else:
        new_bounds = value

    old_bounds = bounds.get(expr, None)

    if old_bounds is not None:
        new_bounds = old_bounds.intersect(
            new_bounds,
            abs_eps=1e-6,
        )
        has_changed = old_bounds != new_bounds
    else:
        has_changed = True
    bounds[expr] = new_bounds
    return has_changed

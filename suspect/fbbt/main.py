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
import pyomo.environ as pyo

from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from pyomo.core.kernel.component_map import ComponentMap

from suspect.interval import Interval
from suspect.fbbt.initialization import initialize_bounds
from suspect.fbbt.propagation import propagate_bounds_leaf_to_root
from suspect.fbbt.tightening import tighten_bounds_root_to_leaf


def perform_fbbt(model, max_iter=10, active=True):
    """Perform FBBT on the model.

    Parameters
    ----------
    model : pyomo.environ.ConcreteModel
        the pyomo model
    max_iter : int
        maximum number of FBBT iterations
    active : boolean
        perform FBBT only on active constraints

    Returns
    -------
    ComponentMap
        tightened bounds for variables and expressions
    """
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")

    bounds = ComponentMap()

    for constraint in model.component_data_objects(pyo.Constraint, active=active, descend_into=True):
        lower = pyo.value(constraint.lower)
        upper = pyo.value(constraint.upper)
        bounds[constraint.body] = Interval(lower, upper)

    for iter in range(max_iter):
        changed = _perform_fbbt_step(model, bounds, iter, active)
        if not changed:
            print('Iter = ', iter)
            break
    print('Iter = ', iter)
    return bounds

nnode = 0
def _perform_fbbt_step(model, bounds, iter, active):
    from timeit import default_timer as timer
    global nnode
    nnode = 0
    if iter == 0:
        func = _initialize_then_propagate_bounds_on_expr
    else:
        func = _tighten_then_propagate_bounds_on_expr

    changed = False
    cchanged_counter = 0
    for objective in model.component_data_objects(pyo.Objective, active=active, descend_into=True):
        s = timer()
        c = func(objective.expr, bounds)
        end = timer()
        if c:
            cchanged_counter += 1
        changed |= c

    for constraint in model.component_data_objects(pyo.Constraint, active=active, descend_into=True):
        s = timer()
        c = func(constraint.body, bounds)
        end = timer()
        if c:
            cchanged_counter += 1
        changed |= c

    print('   CHanged Counter = ', cchanged_counter)
    print('   Nodes visited = ', nnode)
    return changed


def _initialize_then_propagate_bounds_on_expr(expr, bounds):
    return _perform_fbbt_iteration_on_expr(
        expr, bounds, initialize_bounds, propagate_bounds_leaf_to_root
    )


def _tighten_then_propagate_bounds_on_expr(expr, bounds):
    return _perform_fbbt_iteration_on_expr(
        expr, bounds, tighten_bounds_root_to_leaf, propagate_bounds_leaf_to_root
    )


def _perform_fbbt_iteration_on_expr(expr, bounds, tighten, propagate):
    def enter_node(node):
        result = tighten(node, bounds)
        global nnode
        nnode += 1
        if result is None:
            return None, None
        if isinstance(result, list):
            for arg, r in zip(expr.args, result):
                _update_bounds_map(bounds, arg, r)
        else:
            for arg, v in result.items():
                _update_bounds_map(bounds, arg, v)
        return None, None

    def exit_node(node, data):
        result = propagate(node, bounds)
        existing = bounds.get(node, None)
        _update_bounds_map(bounds, node, result)
        if existing is None:
            return True
        return existing != result

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

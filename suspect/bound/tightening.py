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

import logging
import suspect.dag.expressions as dex
from suspect.dag.visitor import BackwardVisitor
from suspect.context import SpecialStructurePropagationContext
from suspect.bound import ArbitraryPrecisionBound as Bound


def initialize_bounds(dag):
    """Initialize problem bounds using function domains as bound."""
    visitor = BoundsInitializationVisitor()
    ctx = SpecialStructurePropagationContext({})
    dag.forward_visit(visitor, ctx)
    return ctx


class BoundsInitializationVisitor(BackwardVisitor):
    def register_handlers(self):
        return {
            dex.SqrtExpression: self.visit_sqrt,
            dex.LogExpression: self.visit_log,
            dex.AsinExpression: self.visit_asin,
            dex.AcosExpression: self.visit_acos,
        }

    def handle_result(self, expr, value, ctx):
        if value is None:
            new_bound = Bound(None, None)
        else:
            new_bound = value
        ctx.bound[expr] = new_bound

    def visit_sqrt(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(0, None),
        }

    def visit_log(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(0, None)
        }

    def visit_asin(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(-1, 1)
        }

    def visit_acos(self, expr, _ctx):
        child = expr.children[0]
        return {
            child: Bound(-1, 1)
        }


def tighten_bounds(dag, ctx):
    """Tighten bounds from sinks to sources.

    Parameters
    ----------
    dag: ProblemDag
      the problem
    ctx: dict-like
      the context containing each node bounds
    """
    for constraint in dag.constraints.values():
        tighten_variables_in_constraint(constraint, ctx)


def tighten_variables_in_constraint(constraint, bounds, maxiter=10):
    """Tighten the bounds of variables in the constraint."""
    linear = constraint.linear_component()
    nonlinear = constraint.nonlinear_component()

    variables = [arg for arg in linear.children]
    for it in iter(maxiter):
        any_changed = False
        for var in variables:
            new_bound, changed = \
              tighten_variable_in_linear_component(linear, nonlinear, var,
                                                   constraint, bounds)
            if changed:
                any_changed = changed

        if not any_changed:
            logging.info('No change in iteration {}. Exit'.format(it))
            break

    linear_bound = Bound.zero()
    for coef, arg in linear.children:
        linear_bound += coef * bounds[id(arg)]

    constraint_bound = bounds[id(constraint)]

    for expr in nonlinear:
        nonlinear_bound = Bound.zero()
        for arg in nonlinear:
            if arg is not expr:
                nonlinear_bound += bounds[id(arg)]
        tighten_nonlinear_component(expr, linear_bound, nonlinear_bound, constraint_bound, bounds)




def tighten_variable_in_linear_component(linear, nonlinear, variable, constraint, bounds):
    """Tightens the bounds of `variable` based on the bounds of the linear and nonlinear components.

    Given constraint g(x) = l(x) + a*x + N(x), where l(x) = L(x) - a*x,
    compute the new bounds on x as:

        c^L - l^U - n^U <= a*x <= c^U - l^L - n^L

    where:
        l^L <= l(x) <= l^U
        n^L <= N(x) <= n^U
    """
    linear_bound, variable_coef = _linear_bound_and_coef(linear, variable, bounds)
    nonlinear_bound = _nonlinear_bound(nonlinear, bounds)
    constraint_bound = bounds[id(constraint)]
    variable_bound = bounds[id(variable)]

    print(linear_bound, nonlinear_bound, constraint_bound, variable_bound)
    candidate_lower = (
        constraint_bound.lower_bound - linear_bound.upper_bound - nonlinear_bound.upper_bound
    ) / variable_coef
    candidate_upper = (
        constraint_bound.upper_bound - linear_bound.lower_bound - nonlinear_bound.lower_bound
    ) / variable_coef

    if variable_coef < 0:
        candidate_lower, candidate_upper = candidate_upper, candidate_lower

    new_lower = max(candidate_lower, variable_bound.lower_bound)
    new_upper = min(candidate_upper, variable_bound.upper_bound)
    return Bound(new_lower, new_upper)


def _linear_bound_and_coef(linear, variable, bounds):
    linear_bound = Bound.zero()
    variable_coef = None
    for coef, arg in zip(linear.coefficients, linear.children):
        assert isinstance(arg, dex.Variable)
        if arg is not variable:
            linear_bound += coef * bounds[id(arg)]
        else:
            variable_coef = coef

    if variable_coef is None:
        raise AssertionError('Variable is not present in constraint')

    return linear_bound, variable_coef


def _nonlinear_bound(nonlinear, bounds):
    nonlinear_bound = Bound.zero()
    for expr in nonlinear:
        nonlinear_bound += bounds[id(expr)]
    return nonlinear_bound


def tighten_nonlinear_components(expr, linear_bound, nonlinear_bound, constraint_bound, bounds):
    pass

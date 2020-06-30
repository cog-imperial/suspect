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
import warnings

import pyomo.environ as pe

from suspect.fbbt import perform_fbbt
from suspect.interval import Interval
from suspect.convexity import Convexity
from suspect.polynomial_degree import polynomial_degree
from suspect.propagation import propagate_special_structure

logger = logging.getLogger('suspect')


class ModelInformation(object):
    """Hold structure information about a problem.

    Attributes
    ----------
    variables : dict
       map variables names to their information
    objectives : dict
       map objectives names to their information
    constraints : dict
       map constraints names to their information
    """
    def __init__(self, name, variables, objectives, constraints):
        self.name = name
        self.variables = variables
        self.objectives = objectives
        self.constraints = constraints

    def num_variables(self):
        """Returns the number of variables in the problem."""
        return len(self.variables)

    def num_binaries(self):
        """Returns the number of binary variables in the problem."""
        binaries = [
            v for v in self.variables.values()
            if v['type'] == 'binary'
        ]
        return len(binaries)

    def num_integers(self):
        """Returns the number of integer variables in the problem."""
        integers = [
            v for v in self.variables.values()
            if v['type'] == 'integer'
        ]
        return len(integers)

    def num_constraints(self):
        """Returns the number of constraints in the problem."""
        return len(self.constraints)

    def conscurvature(self):
        """Returns the convexity of each constraint."""
        return dict(
            (k, v['convexity'])
            for k, v in self.constraints.items()
        )

    def objcurvature(self):
        """Returns the convexity of each objective."""
        return dict(
            (k, v['convexity'])
            for k, v in self.objectives.items()
        )

    def objtype(self):
        """Returns the type of each objective."""
        def _objtype(v):
            cvx = v['convexity']
            deg = v['polynomial_degree']
            if cvx.is_linear():
                assert deg == 1
                return 'linear'
            elif deg == 2:
                return 'quadratic'
            elif deg is not None:
                return 'polynomial'
            else:
                return 'nonlinear'

        return dict(
            (k, _objtype(v))
            for k, v in self.objectives.items()
        )


def detect_special_structure(problem, max_iter=10):
    """Detect special structure in the problem.

    Parameters
    ----------
    model : suspect.dag.ProblemDag or ConcreteModel
        the problem DAG or a Pyomo ConcreteModel.

    max_iter : int
        number of maximum bound propagation/tightening iterations.

    Returns
    -------
    ModelInformation
        an object containing the detected infomation about the problem.
    """
    bounds = perform_fbbt(problem)

    monotonicity, convexity = propagate_special_structure(problem, bounds)

    variables = {}
    for variable in problem.component_data_objects(pe.Var, active=True, descend_into=True):
        if variable.is_binary():
            variable_type = 'binary'
        elif variable.is_integer():
            variable_type = 'integer'
        else:
            variable_type = 'continuous'
        var_bounds = bounds[variable]
        variable_name = variable.name

        variables[variable_name] = {
            'name': variable_name,
            'type': variable_type,
            'lower_bound': var_bounds.lower_bound,
            'upper_bound': var_bounds.upper_bound,
        }

    objectives = {}
    for obj in problem.component_data_objects(pe.Objective, active=True, descend_into=True):
        if obj.is_minimizing():
            sense = 'min'
        else:
            sense = 'max'
        obj_bounds = bounds.get(obj.expr, Interval(None, None))
        cvx = convexity[obj.expr]
        poly = obj.expr.polynomial_degree()

        if sense == 'max':
            cvx = cvx.negate()

        objectives[obj.name] = {
            'sense': sense,
            'convexity': cvx,
            'polynomial_degree': poly,
            'lower_bound': obj_bounds.lower_bound,
            'upper_bound': obj_bounds.upper_bound,
        }

    constraints = {}
    for cons in problem.component_data_objects(pe.Constraint, active=True, descend_into=True):
        if cons.has_lb() and cons.has_ub():
            type_ = 'equality'
        else:
            type_ = 'inequality'

        cons_bounds = bounds.get(cons.body, Interval(None, None))
        cvx = convexity[cons.body]
        poly = cons.body.polynomial_degree()

        if type_ == 'equality':
            if not cvx.is_linear():
                cvx = Convexity.Unknown
        else:
            if cons.has_lb():
                cvx = cvx.negate()

        constraints[cons.name] = {
            'type': type_,
            'convexity': cvx,
            'polynomial_degree': poly,
            'lower_bound': cons_bounds.lower_bound,
            'upper_bound': cons_bounds.upper_bound,
        }

    return ModelInformation(
        name=problem.name,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
    )

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

import warnings
import logging
from suspect.pyomo.convert import dag_from_pyomo_model
from suspect.fbbt import BoundsTightener, FBBTStopCriterion
from suspect.dag.iterator import DagForwardIterator, DagBackwardIterator
from suspect.context import SpecialStructurePropagationContext
from suspect.interval import Interval
from suspect.propagation import propagate_special_structure
from suspect.polynomial_degree import polynomial_degree
from pyomo.environ import ConcreteModel # pylint: disable=no-name-in-module,import-error


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
                assert deg.is_linear()
                return 'linear'
            elif deg.is_quadratic():
                return 'quadratic'
            elif deg.is_polynomial():
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
    if isinstance(problem, ConcreteModel):
        problem = dag_from_pyomo_model(problem)

    ctx = SpecialStructurePropagationContext()

    bounds_tightener = BoundsTightener(DagForwardIterator(), DagBackwardIterator(), FBBTStopCriterion())
    bounds_tightener.tighten(problem, ctx)

    polynomial = polynomial_degree(problem, ctx.polynomial)
    monotonicity, convexity = propagate_special_structure(problem, ctx)

    variables = {}
    for variable_name, variable in problem.variables.items():
        if variable.is_binary():
            variable_type = 'binary'
        elif variable.is_integer():
            variable_type = 'integer'
        else:
            variable_type = 'continuous'
        var_bounds = ctx.bound[variable]

        if variable.name in variables:
            warnings.warn('Duplicate variable {}'.format(variable.name))

        variables[variable_name] = {
            'name': variable_name,
            'type': variable_type,
            'lower_bound': var_bounds.lower_bound,
            'upper_bound': var_bounds.upper_bound,
        }

    objectives = {}
    for obj_name, obj in problem.objectives.items():
        if obj_name in objectives:
            warnings.warn('Duplicate objective {}'.format(obj_name))

        if obj.is_minimizing():
            sense = 'min'
        else:
            sense = 'max'
        obj_bounds = ctx.bound.get(obj, Interval(None, None))
        cvx = ctx.convexity[obj]
        poly = polynomial[obj]

        objectives[obj.name] = {
            'sense': sense,
            'convexity': cvx,
            'polynomial_degree': poly,
            'lower_bound': obj_bounds.lower_bound,
            'upper_bound': obj_bounds.upper_bound,
        }

    constraints = {}
    for cons_name, cons in problem.constraints.items():
        if cons_name in constraints:
            warnings.warn('Duplicate constraint {}'.format(cons_name))

        if cons.is_equality():
            type_ = 'equality'
        else:
            type_ = 'inequality'

        cvx = ctx.convexity[cons]
        poly = polynomial[cons]

        constraints[cons_name] = {
            'type': type_,
            'convexity': cvx,
            'polynomial_degree': poly,
        }

    return ModelInformation(
        name=problem.name,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
    )

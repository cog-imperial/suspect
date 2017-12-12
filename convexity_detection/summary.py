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
from convexity_detection.bounds import expression_bounds
from convexity_detection.convexity import expression_convexity
from convexity_detection.tightening import tighten_model_bounds
from convexity_detection.util import (
    model_variables,
    model_constraints,
    model_objectives,
    bounds_and_expr,
)


class ModelInformation(object):
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
            elif deg is None:
                return 'nonlinear'
            else:
                return 'polynomial'

        return dict(
            (k, _objtype(v))
            for k, v in self.objectives.items()
        )


def detect_special_structure(model):
    """Detect special structure in the model.

    Parameters
    ----------
    model: Model
        a Pyomo model

    Returns
    -------
    ModelInformation
        an object containing the detected infomation about the problem
    """
    bounds = tighten_model_bounds(model)

    variables = {}
    for variable in model_variables(model):
        if variable.is_binary():
            variable_type = 'binary'
        elif variable.is_integer():
            variable_type = 'integer'
        else:
            variable_type = 'continuous'
        var_bounds = bounds[variable]

        if variable.name in variables:
            warnings.warn('Duplicate variable {}'.format(variable.name))

        variables[variable.name] = {
            'name': variable.name,
            'type': variable_type,
            'lower_bound': var_bounds.l,
            'upper_bound': var_bounds.u,
        }

    objectives = {}
    for obj in model_objectives(model):
        if obj.name in objectives:
            warnings.warn('Duplicate objective {}'.format(obj.name))

        if obj.is_minimizing():
            sense = 'min'
            expr = obj.expr
            cvx = expression_convexity(expr, bounds)
        else:
            sense = 'max'
            expr = -obj.expr
            cvx = expression_convexity(expr, bounds)
            cvx = cvx.negate()

        obj_bounds = expression_bounds(obj.expr, bounds)

        objectives[obj.name] = {
            'sense': sense,
            'convexity': cvx,
            'polynomial_degree': obj.expr.polynomial_degree(),
            'lower_bound': obj_bounds.l,
            'upper_bound': obj_bounds.u,
        }

    constraints = {}
    for cons in model_constraints(model):
        if cons.name in constraints:
            warnings.warn('Duplicate constraint {}'.format(cons.name))

        cons_bounds, expr = bounds_and_expr(cons.expr)
        if cons.equality:
            type_ = 'equality'
            bound = cons_bounds.u
            cvx1 = expression_convexity(expr <= bound, bounds)
            cvx2 = expression_convexity(-expr <= - bound, bounds)
            cvx = cvx1.combine(cvx2)
        else:
            type_ = 'inequality'
            cvx1 = expression_convexity(expr <= cons_bounds.u, bounds)
            cvx2 = expression_convexity(-expr <= -cons_bounds.l, bounds)
            cvx = cvx1.combine(cvx2)

        constraints[cons.name] = {
            'type': type_,
            'convexity': cvx,
            'polynomial_degree': cons.body.polynomial_degree(),
        }

    return ModelInformation(
        name=model.name,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
    )

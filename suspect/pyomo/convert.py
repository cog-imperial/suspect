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

from copy import deepcopy
from numbers import Number
import pyomo.environ as aml
import pyomo.core.expr.expr_pyomo5 as pex
from pyomo.core.expr.expr_pyomo5 import (
    ExpressionValueVisitor,
    nonpyomo_leaf_types,
    NumericConstant,
)
from suspect.pyomo.util import (
    model_variables,
    model_objectives,
    model_constraints,
    bounds_and_expr,
)
from suspect.pyomo.expr_dict import ExpressionDict
from suspect.float_hash import BTreeFloatHasher
from suspect.dag.expressions import create_wrapper_node_with_local_data
from suspect.dag.dag import ProblemDag
import suspect.dag.expressions as dex


def dag_from_pyomo_model(model):
    """Convert the Pyomo ``model`` to SUSPECT DAG.

    Parameters
    ----------
    model : ConcreteModel
        the Pyomo model.

    Returns
    -------
    ProblemDag
        SUSPECT problem DAG.
    """
    dag = ProblemDag(name=model.name)
    factory = ComponentFactory(dag)
    for omo_var in model_variables(model):
        new_var = factory.variable(omo_var)
        dag.add_variable(new_var)

    for omo_cons in model_constraints(model):
        new_cons = factory.constraint(omo_cons)
        dag.add_constraint(new_cons)

    for omo_obj in model_objectives(model):
        new_obj = factory.objective(omo_obj)
        dag.add_objective(new_obj)

    return dag


class ComponentFactory(object):
    def __init__(self, dag):
        self.dag = dag
        self._components = ExpressionDict(float_hasher=BTreeFloatHasher())
        self._visitor = _ConvertExpressionVisitor(self._components, self.dag)

    def variable(self, omo_var):
        comp = self._components.get(omo_var)
        if comp is not None:
            return comp
        new_var = deepcopy(omo_var)
        self._components[omo_var] = new_var
        return new_var

    def constraint(self, omo_cons):
        bounds, expr = bounds_and_expr(omo_cons.expr)
        new_expr = self.expression(expr)
        constraint = dex.Constraint(
            omo_cons.name,
            bounds.lower_bound,
            bounds.upper_bound,
            [new_expr],
        )
        return constraint

    def objective(self, omo_obj):
        if omo_obj.is_minimizing():
            sense = dex.Sense.MINIMIZE
        else:
            sense = dex.Sense.MAXIMIZE

        new_expr = self.expression(omo_obj.expr)
        obj = dex.Objective(
            omo_obj.name, sense=sense, children=[new_expr]
        )
        return obj

    def expression(self, expr):
        return self._visitor.dfs_postorder_stack(expr)


class _ConvertExpressionVisitor(ExpressionValueVisitor):
    def __init__(self, memo, dag):
        self.memo = memo
        self.dag = dag

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            expr = NumericConstant(float(node))
            self.set(expr, expr)
            return True, expr

        if node.is_constant():
            self.set(expr, expr)
            return True, expr

        if node.is_variable_type():
            var = self.get(node)
            assert var is not None
            return True, var
        return False, None

    def visit(self, node, values):
        if self.get(node) is not None:
            return self.get(node)

        new_expr = create_wrapper_node_with_local_data(node, tuple(values))
        self.set(node, new_expr)
        return new_expr

    def get(self, expr):
        if isinstance(expr, Number):
            const = aml.NumericConstant(expr)
            return self.get(const)
        else:
            return self.memo[expr]

    def set(self, expr, new_expr):
        if self.memo.get(expr) is not None:
            return
        self.memo[expr] = new_expr
        self.dag.add_vertex(new_expr)

from numbers import Number

import pyomo.environ as pyo
from pyomo.repn.standard_repn import generate_standard_repn
from pyomo.core.expr.numeric_expr import (
    nonpyomo_leaf_types,
    SumExpression,
    LinearExpression,
)
from pyomo.core.expr.numvalue import NumericConstant
from pyomo.core.expr.visitor import ExpressionValueVisitor
from pyomo.common.collections import ComponentMap

from suspect.float_hash import BTreeFloatHasher
from suspect.pyomo.expr_dict import ExpressionDict
from suspect.pyomo.quadratic import QuadraticExpression


def create_connected_model(model, active=True, connect_max_linear_children=50, connect_max_quadratic_children=100):
    connected = model.clone()

    model_to_connected_map = ComponentMap()
    components = ExpressionDict(float_hasher=BTreeFloatHasher())

    for var in model.component_data_objects(pyo.Var, active=active, sort=True, descend_into=True):
        connected_var = connected.find_component(var)
        model_to_connected_map[var] = connected_var
        components[connected_var] = connected_var

    for constraint in model.component_data_objects(pyo.Constraint, active=active, sort=True, descend_into=True):
        connected_constraint = connected.find_component(constraint)
        model_to_connected_map[constraint] = connected_constraint

    for objective in model.component_data_objects(pyo.Objective, active=active, sort=True, descend_into=True):
        connected_objective = connected.find_component(objective)
        model_to_connected_map[objective] = connected_objective

    convert_visitor = _ConvertExpressionVisitor(
        components, model_to_connected_map, connect_max_linear_children, connect_max_quadratic_children
    )

    for constraint in connected.component_data_objects(pyo.Constraint, active=active, sort=True, descend_into=True):
        new_body = convert_visitor.dfs_postorder_stack(constraint.body)
        constraint._body = new_body

    for objective in connected.component_data_objects(pyo.Objective, active=active, sort=True, descend_into=True):
        new_body = convert_visitor.dfs_postorder_stack(objective.expr)
        objective._expr = new_body

    return connected, model_to_connected_map


class _ConvertExpressionVisitor(ExpressionValueVisitor):
    def __init__(self, memo, component_map, connect_max_linear_children, connect_max_quadratic_children):
        self.memo = memo
        self.component_map = component_map
        self.connect_max_linear_children = connect_max_linear_children
        self.connect_max_quadratic_children = connect_max_quadratic_children

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            expr = NumericConstant(float(node))
            expr = self.set(expr, expr)
            return True, expr

        if node.is_constant():
            expr = self.set(node, node)
            return True, expr

        if node.is_variable_type():
            var = self.get(node)
            assert var is not None
            return True, var

        if _is_quadratic_expression(node) and node.nargs() > self.connect_max_quadratic_children:
            return True, node

        if _is_linear_expression(node) and node.nargs() > self.connect_max_linear_children:
            return True, node

        return False, None

    def visit(self, node, values):
        new_expr = self.get(node)
        if new_expr is not None:
            self.set(node, new_expr)
            return new_expr

        if _is_quadratic_expression(node):
            new_expr = _convert_quadratic_expression(node)
        else:
            if node.is_named_expression_type():
                assert len(values) == 1
                node.expr = values[0]
                new_expr = node
            else:
                new_expr = node.create_node_with_local_data(tuple(values))
        self.set(node, new_expr)
        return new_expr

    def get(self, expr):
        if isinstance(expr, Number):
            const = NumericConstant(expr)
            return self.get(const)
        else:
            return self.memo[expr]

    def set(self, expr, new_expr):
        self.component_map[expr] = new_expr
        if self.memo.get(expr) is not None:
            return self.memo[expr]
        self.memo[expr] = new_expr
        return new_expr


def _is_quadratic_expression(node):
    return type(node) == SumExpression and node.polynomial_degree() == 2 and node.nargs() > 1


def _is_linear_expression(node):
    return (
        type(node) in (SumExpression, LinearExpression)
        and node.polynomial_degree() == 1
    )


def _convert_quadratic_expression(expr):
    # Check if there is any non bilinear term
    repn_result = generate_standard_repn(expr, compute_values=False)
    assert repn_result.quadratic_vars
    assert not repn_result.nonlinear_expr
    quadratic_expr = 0.0

    for (v1, v2), coeff in zip(repn_result.quadratic_vars, repn_result.quadratic_coefs):
        quadratic_expr += (v1 * v2) * float(coeff)

    if not repn_result.linear_vars:
        return QuadraticExpression(quadratic_expr)

    linear_expr = repn_result.constant

    for linear_var, linear_coeff in zip(repn_result.linear_vars, repn_result.linear_coefs):
        linear_expr += linear_var * float(linear_coeff)

    quadratic = QuadraticExpression(quadratic_expr)
    return SumExpression([quadratic, linear_expr])

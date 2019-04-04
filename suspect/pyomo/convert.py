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
        domain = _convert_domain(omo_var.domain)
        new_var = dex.Variable(omo_var.name, omo_var.lb, omo_var.ub, domain)
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
        new_expr.add_parent(constraint)
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
        new_expr.add_parent(obj)
        return obj

    def expression(self, expr):
        return self._visitor.dfs_postorder_stack(expr)



_unary_func_name_to_expr_cls = {
    'sqrt': dex.SqrtExpression,
    'exp': dex.ExpExpression,
    'log': dex.LogExpression,
    'sin': dex.SinExpression,
    'cos': dex.CosExpression,
    'tan': dex.TanExpression,
    'asin': dex.AsinExpression,
    'acos': dex.AcosExpression,
    'atan': dex.AtanExpression,
}


def _convert_as(expr_cls):
    return lambda _, v: expr_cls(v)


def _convert_unary_function(node, values):
    assert len(values) == 1
    expr_cls = _unary_func_name_to_expr_cls.get(node.getname(), None)
    if expr_cls is None:
        raise RuntimeError('Unknown UnaryFunctionExpression type {}'.format(node.getname()))
    return expr_cls(values)


def _is_product_with_reciprocal(children):
    if len(children) != 2:
        return False
    a, b = children
    if isinstance(a, dex.DivisionExpression):
        if isinstance(a.children[0], dex.Constant):
            return a.children[0].value == 1.0
        return False
    if isinstance(b, dex.DivisionExpression):
        if isinstance(b.children[0], dex.Constant):
            return b.children[0].value == 1.0
    return False


def _is_bilinear_product(children):
    if len(children) != 2:
        return False
    a, b = children
    if isinstance(a, dex.Variable) and isinstance(b, dex.Variable):
        return True
    if isinstance(a, dex.Variable) and isinstance(b, ex.LinearExpression):
        return len(b.children) == 1 and b.constant_term == 0.0
    if isinstance(a, dex.LinearExpression) and isinstance(b, dex.Variable):
        return len(a.children) == 1 and a.constant_term == 0.0
    return False


def _bilinear_variables_with_coefficient(children):
    assert len(children) == 2
    a, b = children
    if isinstance(a, dex.Variable) and isinstance(b, dex.Variable):
        return a, b, 1.0
    if isinstance(a, dex.Variable) and isinstance(b, dex.LinearExpression):
        assert len(b.children) == 1
        assert b.constant_term == 0.0
        vb = b.children[0]
        return a, vb, b.coefficient(vb)
    if isinstance(a, dex.LinearExpression) and isinstance(b, dex.Variable):
        assert len(a.children) == 1
        assert a.constant_term == 0.0
        va = a.children[0]
        return va, b, a.coefficient(va)


def _reciprocal_product_numerator_denominator(children):
    a, b = children
    if isinstance(a, dex.DivisionExpression):
        assert not isinstance(b, dex.DivisionExpression)
        _, d = a.children
        return b, d

    assert isinstance(b, dex.DivisionExpression)
    _, d = b.children
    return a, d


def _convert_product(_node, values):
    if _is_product_with_reciprocal(values):
        n, d = _reciprocal_product_numerator_denominator(values)
        return dex.DivisionExpression([n, d])

    if _is_bilinear_product(values):
        a, b, c = _bilinear_variables_with_coefficient(values)
        return dex.QuadraticExpression([a], [b], [c], [a, b])

    return dex.ProductExpression(values)


def _convert_reciprocal(_node, values):
    assert len(values) == 1
    return dex.DivisionExpression([dex.Constant(1.0), values[0]])


def _is_monomial_like_expression(expr):
    if isinstance(expr, dex.Variable):
        return True
    if isinstance(expr, dex.Constant):
        return True
    if isinstance(expr, dex.LinearExpression):
        return True
    return False


def _convert_sum(_node, values):
    # convert sum of variables, monomial terms and constants
    # to linear expression
    if all([_is_monomial_like_expression(v) for v in values]):
        coef_map = {}
        constant = 0.0
        for value in values:
            if isinstance(value, dex.Constant):
                constant += value.value

            if isinstance(value, dex.Variable):
                if value not in coef_map:
                    coef_map[value] = 0.0
                coef_map[value] += 1.0

            if isinstance(value, dex.LinearExpression):
                for var in value.children:
                    if var not in coef_map:
                        coef_map[var] = 0.0
                    coef_map[var] += value.coefficient(var)

        variables = list(coef_map.keys())
        coefficients = [coef_map[v] for v in variables]
        return dex.LinearExpression(coefficients, variables, constant)

    return dex.SumExpression(values)


def _convert_monomial(_node, values):
    const, var = values
    assert isinstance(const, dex.Constant)
    assert isinstance(var, dex.Variable)
    return dex.LinearExpression([const.value], [var], 0.0)


_convert_expr_map = dict()
_convert_expr_map[pex.UnaryFunctionExpression] = _convert_unary_function
_convert_expr_map[pex.ProductExpression] = _convert_product
_convert_expr_map[pex.ReciprocalExpression] = _convert_reciprocal
_convert_expr_map[pex.PowExpression] = _convert_as(dex.PowExpression)
_convert_expr_map[pex.SumExpression] = _convert_sum
_convert_expr_map[pex.MonomialTermExpression] = _convert_monomial
_convert_expr_map[pex.AbsExpression] = _convert_as(dex.AbsExpression)
_convert_expr_map[pex.NegationExpression] = _convert_as(dex.NegationExpression)


class _ConvertExpressionVisitor(ExpressionValueVisitor):
    def __init__(self, memo, dag):
        self.memo = memo
        self.dag = dag

    def visiting_potential_leaf(self, node):
        if node.__class__ in nonpyomo_leaf_types:
            expr = NumericConstant(float(node))
            const = dex.Constant(float(node))
            self.set(expr, const)
            return True, const

        if node.is_variable_type():
            if node.is_fixed():
                expr = NumericConstant(float(node))
                const = dex.Constant(float(node))
                self.set(expr, const)
                return True, const
            return True, self.memo[node]
        return False, None

    def visit(self, node, values):
        if self.memo[node] is not None:
            return self.memo[node]

        print(node, node.__class__, values)

        callback = _convert_expr_map.get(type(node), None)
        if callback is None:
            raise RuntimeError('Unknown expression type {}'.format(type(node)))

        new_expr = callback(node, values)
        self.set(node, new_expr)
        return new_expr

    def get(self, expr):
        if isinstance(expr, Number):
            const = aml.NumericConstant(expr)
            return self.get(const)
        else:
            return self.memo[expr]

    def set(self, expr, new_expr):
        self.memo[expr] = new_expr
        if len(new_expr.children) > 0:
            for child in new_expr.children:
                child.add_parent(new_expr)
        self.dag.add_vertex(new_expr)

    def visit_number(self, n):
        const = aml.NumericConstant(n)
        self.visit_numeric_constant(const)

    def visit_numeric_constant(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        const = dex.Constant(expr.value)
        self.set(expr, const)

    def visit_variable(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]
        raise AssertionError('Unknown variable encountered')

    def visit_product(self, expr):

        if self.memo[expr] is not None:
            return self.memo[expr]
        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        if _is_bilinear(children):
            a, b, c = _bilinear_variables_with_coefficient(children)
            new_expr = dex.QuadraticExpression([a], [b], [c], [a, b])
        else:
            new_expr = dex.ProductExpression(children)
        self.set(expr, new_expr)

    def visit_division(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.DivisionExpression(children)
        self.set(expr, new_expr)

    def visit_sum(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.SumExpression(children)
        self.set(expr, new_expr)

    def visit_linear(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        coeffs = [expr._coef[id(a)] for a in expr._args]
        const = expr._const
        new_expr = dex.LinearExpression(
            coeffs, children, const
        )
        self.set(expr, new_expr)

    def visit_negation(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.NegationExpression(children)
        self.set(expr, new_expr)

    def visit_unary_function(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        assert len(children) == 1
        fun = expr.name
        ExprClass = {
            'sqrt': dex.SqrtExpression,
            'exp': dex.ExpExpression,
            'log': dex.LogExpression,
            'sin': dex.SinExpression,
            'cos': dex.CosExpression,
            'tan': dex.TanExpression,
            'asin': dex.AsinExpression,
            'acos': dex.AcosExpression,
            'atan': dex.AtanExpression,
        }.get(fun)
        if ExprClass is None:
            raise AssertionError('Unknwon function', fun)
        new_expr = ExprClass(children)
        self.set(expr, new_expr)


    def visit_abs(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.AbsExpression(children)
        self.set(expr, new_expr)

    def visit_pow(self, expr):
        if self.memo[expr] is not None:
            return self.memo[expr]

        self._check_children(expr)
        children = [self.get(a) for a in expr._args]
        new_expr = dex.PowExpression(children)
        self.set(expr, new_expr)


def _convert_domain(dom):
    if isinstance(dom, aml.RealSet):
        return dex.Domain.REALS
    elif isinstance(dom, aml.IntegerSet):
        return dex.Domain.INTEGERS
    elif isinstance(dom, aml.BooleanSet):
        return dex.Domain.BINARY

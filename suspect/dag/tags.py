# Copyright 2019 Francesco Ceccon
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

"""Tags are a way to tag a group of nodes as special structure."""
from pyomo.common.collections import ComponentMap
import pyomo.environ as pe
from suspect.pyomo.expressions import nonpyomo_leaf_types
import suspect.pyomo.expressions as dex


class Tag(object):
    """A Tag represents a special structure expression that spans
    across multiple nodes.
    """

    @staticmethod
    def from_expr(problem, expr):
        """Create `Tag` from `expr`.

        Parameters
        ----------
        problem : dag
            the problem containing the expression
        expr : expression
            the root expression

        Returns
        -------
        Tag or None
            returns a new Tag if expression matches, otherwise None
        """
        raise NotImplementedError('Tag.from_expr')


class LinearExpressionTag(Tag):
    def __init__(self, problem, expr, coef_map, constant):
        self.problem = problem
        self.expr = expr
        self._coef_map = coef_map
        self.constant = constant

    @staticmethod
    def from_expr(problem, expr):
        if not isinstance(expr, dex.SumExpression):
            return None

        if not all(LinearExpressionTag._is_linear_type(a)
                   for a in expr.args):
            return None

        coef_map = ComponentMap()
        constant = 0.0
        for arg in expr.args:
            if LinearExpressionTag._is_var(arg):
                if arg not in coef_map:
                    coef_map[arg] = 0.0
                coef_map[arg] += 1.0
            elif LinearExpressionTag._is_monomial(arg):
                assert len(arg.args) == 2
                const, var = arg.args
                if var not in coef_map:
                    coef_map[var] = 0.0
                coef_map[var] += pe.value(const)
            else:
                assert LinearExpressionTag._is_constant(arg)
                constant += pe.value(arg)
        return LinearExpressionTag(problem, expr, coef_map, constant)

    def coefficient(self, var):
        return self._coef_map[var]

    @staticmethod
    def _is_linear_type(expr):
        return (
            LinearExpressionTag._is_monomial(expr) or
            LinearExpressionTag._is_var(expr) or
            LinearExpressionTag._is_constant(expr)
        )

    @staticmethod
    def _is_monomial(expr):
        return isinstance(expr, dex.MonomialTermExpression)

    @staticmethod
    def _is_var(expr):
        return expr.is_variable_type()

    @staticmethod
    def _is_constant(expr):
        return expr.is_constant()



class QuadraticExpressionTag(Tag):
    def __init__(self, problem, expr, coef_map):
        self.problem = problem
        self.expr = expr
        self._coef_map = coef_map

    @staticmethod
    def from_expr(problem, expr):
        if not isinstance(expr, dex.SumExpression):
            return None

        if not all(a.polynomial_degree() == 2 for a in expr.args):
            return None

        # build bilinear terms as sparse matrix
        return None

    @staticmethod
    def _vars_and_coef(self, expr):
        pass

    # cases:
    # x ** 2
    # xy
    # (3x)y
    # (xy)3
    # x(3y)
    def terms(self):
        pass

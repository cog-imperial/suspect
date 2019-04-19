# Copyright 2018 Francesco Ceccon
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

"""Convexity detector for L2-Norm expressions."""
from suspect.ext import ConvexityDetector
from suspect.convexity import Convexity
from suspect.interfaces import UnaryFunctionRule
from suspect.expression import ExpressionType, UnaryFunctionType


class L2NormConvexityDetector(ConvexityDetector):
    """Detect L_2 Norm convexity in the form

       sqrt(eps + sum_i(x_i**2))
    """
    def __init__(self, _problem):
        super().__init__()

    def visit_expression(self, expr, convexity, mono, bounds):
        return False, None
        return {
            ExpressionType.UnaryFunction: L2NormRule(),
        }


class L2NormRule(UnaryFunctionRule):
    """Detect L_2 Norm expressions."""
    func_type = UnaryFunctionType.Sqrt

    def apply(self, expr, _ctx):
        child = expr.children[0]
        if child.expression_type != ExpressionType.Sum:
            return None

        grandchildren = child.children
        if all([self._square_or_constant(child) for child in grandchildren]):
            return Convexity.Convex
        return None

    def _square_or_constant(self, expr):
        expr_type = expr.expression_type
        if expr_type == ExpressionType.Constant:
            return True
        if expr_type == ExpressionType.Power:
            base, expo = expr.children
            base_is_variable = base.expression_type == ExpressionType.Variable
            expo_is_variable = expo.expression_type == ExpressionType.Constant
            if base_is_variable and expo_is_variable:
                return expo.value == 2
            return False
        if expr_type == ExpressionType.Product:
            if len(expr.children) != 2:
                return False
            f, g = expr.children
            return f is g and f.expression_type == ExpressionType.Variable
        return False

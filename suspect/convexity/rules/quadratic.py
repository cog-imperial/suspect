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

"""Convexity detection rules for quadratic expressions."""
import numpy as np

from suspect.convexity.convexity import Convexity
from suspect.convexity.rules.rule import ConvexityRule


def _gershgorin_circles_test(expr, var_to_idx):
    """Check convexity by computing Gershgorin circles without building
    the coefficients matrix.

    If the circles lie in the nonnegative (nonpositive) space, then the matrix
    is positive (negative) definite.

    Parameters
    ----------
    expr : QuadraticExpression
        the quadratic expression
    var_to_idx : dict-like
        map a var to a numerical index between 0 and n, where n is the number
        of args of expr

    Returns
    -------
    Convexity if the expression is Convex or Concave, None otherwise.
    """
    n = expr.nargs()
    row_circles = np.zeros(n)
    diagonal = np.zeros(n)

    for term in expr.terms:
        i = var_to_idx[term.var1]
        j = var_to_idx[term.var2]
        if i == j:
            diagonal[i] = term.coefficient
        else:
            coef = np.abs(term.coefficient / 2.0)
            row_circles[j] += coef
            row_circles[i] += coef

    if np.all((diagonal - row_circles) >= 0):
        return Convexity.Convex
    if np.all((diagonal + row_circles) <= 0):
        return Convexity.Concave
    return None


def _eigval_test(A):
    """Check convexity by computing eigenvalues.

    Parameters
    ----------
    A : matrix
        the (symmetric) coefficients matrix

    Returns
    -------
    Convexity if the expression is Convex or Concave, None otherwise.
    """
    eigv = np.linalg.eigvalsh(A)
    if np.all(eigv >= 0):
        return Convexity.Convex
    elif np.all(eigv <= 0):
        return Convexity.Concave
    return None


class QuadraticRule(ConvexityRule):
    """Return convexity of quadratic."""
    def __init__(self, max_expr_children=None):
        if max_expr_children is None:
            max_expr_children = 100
        self.max_expr_children = max_expr_children

    def apply(self, expr, convexity, mono, bounds):
        var_to_idx = dict([(v, i) for i, v in enumerate(expr.args)])
        convexity = _gershgorin_circles_test(expr, var_to_idx)

        if convexity is not None:
            return convexity

        # Build coefficient matrix
        n = len(expr.args)
        if self.max_expr_children and n > self.max_expr_children:
            return Convexity.Unknown

        A = np.zeros((n, n))
        for term in expr.terms:
            i = var_to_idx[term.var1]
            j = var_to_idx[term.var2]
            if term.var1 == term.var2:
                A[i, j] = term.coefficient
            else:
                A[i, j] = term.coefficient / 2.0
                A[j, i] = A[i, j]

        convexity = _eigval_test(A)
        if convexity is not None:
            return convexity

        return Convexity.Unknown

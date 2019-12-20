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


def _gershgorin_circles_test(A):
    """Check convexity by computing Gershgorin circles.

    If the circles lie in the nonnegative (nonpositive) space, then the matrix
    is positive (negative) definite.

    Parameters
    ----------
    A : matrix
        the (symmetric) coefficients matrix

    Returns
    -------
    Convexity if the expression is Convex or Concave, None otherwise.
    """
    # Check both row and column circles for each eigenvalue
    diag = np.diag(A)
    abs_A = np.abs(A)
    abs_diag = np.abs(diag)
    row_circles = np.sum(abs_A, 1) - abs_diag
    col_circles = np.sum(abs_A, 0) - abs_diag
    circles = np.minimum(row_circles, col_circles)
    if np.all((diag - circles) >= 0):
        return Convexity.Convex
    if np.all((diag + circles) <= 0):
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
    def __init__(self, max_matrix_size=None):
        if max_matrix_size is None:
            max_matrix_size = 100
        self.max_matrix_size = max_matrix_size

    def apply(self, expr, convexity, mono, bounds):
        # Build coefficient matrix
        n = len(expr.args)
        if self.max_matrix_size and n > self.max_matrix_size:
            return Convexity.Unknown

        A = np.zeros((n, n))
        var_to_idx = dict([(v, i) for i, v in enumerate(expr.args)])
        for term in expr.terms:
            i = var_to_idx[term.var1]
            j = var_to_idx[term.var2]
            if term.var1 == term.var2:
                A[i, j] = term.coefficient
            else:
                A[i, j] = term.coefficient / 2.0
                A[j, i] = A[i, j]

        convexity = _gershgorin_circles_test(A)

        if convexity is not None:
            return convexity

        convexity = _eigval_test(A)
        if convexity is not None:
            return convexity

        return Convexity.Unknown

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



class QuadraticRule(ConvexityRule):
    """Return convexity of quadratic."""
    def __init__(self, max_matrix_size=None):
        if max_matrix_size is None:
            max_matrix_size = 100
        self.max_matrix_size = max_matrix_size

    def apply(self, expr, convexity, mono, bounds):
        # Sum of squares
        if self._is_sum_of_squares(expr):
            coefficients = np.array([term.coefficient for term in expr.terms])
            if np.all(coefficients >= 0):
                return Convexity.Convex
            elif np.all(coefficients <= 0):
                return Convexity.Concave
            else:
                return Convexity.Unknown

        # try compute eigvalues
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
        eigv = np.linalg.eigvalsh(A)
        if np.all(eigv >= 0):
            return Convexity.Convex
        elif np.all(eigv <= 0):
            return Convexity.Concave
        return Convexity.Unknown

    def _is_sum_of_squares(self, expr):
        if len(expr.terms) == 0:
            return False
        for term in expr.terms:
            if term.var1 != term.var2:
                return False
        return True

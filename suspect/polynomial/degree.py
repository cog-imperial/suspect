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

"""Polynomial degree class."""


class PolynomialDegree(object):
    """Represent a polynomial degree."""
    def __init__(self, degree):
        self.degree = degree

    @classmethod
    def not_polynomial(cls):
        """Constructor for non polynomial polynomial."""
        return PolynomialDegree(None)

    def is_polynomial(self):
        """Predicate to check if it's polynomial."""
        return self.degree is not None

    def is_constant(self):
        """Predicate to check if it's constant."""
        return self.is_polynomial() and self.degree == 0

    def is_linear(self):
        """Predicate to check if it's linear."""
        return self.is_polynomial() and self.degree == 1

    def is_quadratic(self):
        """Predicate to check if it's quadratic."""
        return self.is_polynomial() and self.degree == 2

    def __add__(self, other):
        if self.is_polynomial() and other.is_polynomial():
            return PolynomialDegree(self.degree + other.degree)
        return self.not_polynomial()

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return PolynomialDegree(self.degree * other)
        return PolynomialDegree.not_polynomial()

    def __gt__(self, other):
        if not self.is_polynomial():
            return True
        if not other.is_polynomial():
            return False
        return self.degree > other.degree

    def __eq__(self, other):
        return self.degree == other.degree

    def __str__(self):
        return 'PolynomialDegree(degree={})'.format(self.degree)

    def __repr__(self):
        return '<{} at {}>'.format(str(self), hex(id(self)))

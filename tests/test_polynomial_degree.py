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

import pytest
from tests.conftest import PlaceholderExpression
import suspect.dag.expressions as dex
from suspect.polynomial_degree import PolynomialDegreeVisitor, PolynomialDegree


@pytest.fixture
def visitor():
    return PolynomialDegreeVisitor()


class PolynomialTestContext(object):
    def __init__(self):
        self.polynomial = {}

    def __getitem__(self, key):
        return self.polynomial[key]

    def __setitem__(self, key, value):
        self.polynomial[key] = value


@pytest.fixture
def ctx():
    return PolynomialTestContext()


def test_degree_of_variable(visitor, ctx):
    v = dex.Variable('x0', None, None)
    visitor(v, ctx)
    assert ctx[v].degree == 1


def test_degree_of_constant(visitor, ctx):
    c = dex.Constant(1)
    visitor(c, ctx)
    assert ctx[c].degree == 0


def test_degree_of_constraint(visitor, ctx):
    p = PlaceholderExpression()
    ctx[p] = PolynomialDegree(2)
    c = dex.Constraint('c', None, None, [p])
    visitor(c, ctx)
    assert ctx[c].degree == 2


def test_degree_of_objective(visitor, ctx):
    p = PlaceholderExpression()
    ctx[p] = PolynomialDegree(2)
    c = dex.Objective('o', children=[p])
    visitor(c, ctx)
    assert ctx[c].degree == 2


def test_degree_of_polynomial_product(visitor, ctx):
    children = [PlaceholderExpression() for _ in range(5)]
    for i, c in enumerate(children):
        ctx[c] = PolynomialDegree(i)
    p = dex.ProductExpression(children)
    visitor(p, ctx)
    assert ctx[p].degree == 10


def test_degree_of_non_polynomial_product(visitor, ctx):
    children = [PlaceholderExpression() for _ in range(5)]
    for i, c in enumerate(children):
        if i == 0:
            ctx[c] = PolynomialDegree.not_polynomial()
        else:
            ctx[c] = PolynomialDegree(i)
    p = dex.ProductExpression(children)
    visitor(p, ctx)
    assert not ctx[p].is_polynomial()


def test_degree_of_division_constant_denominator(visitor, ctx):
    num = PlaceholderExpression()
    den = PlaceholderExpression()
    d = dex.DivisionExpression([num, den])
    ctx[num] = PolynomialDegree(2)
    ctx[den] = PolynomialDegree(0)
    visitor(d, ctx)
    assert ctx[d].degree == 2


def test_degree_of_division_non_constant_denominator(visitor, ctx):
    num = PlaceholderExpression()
    den = PlaceholderExpression()
    d = dex.DivisionExpression([num, den])
    ctx[num] = PolynomialDegree(2)
    ctx[den] = PolynomialDegree.not_polynomial()
    visitor(d, ctx)
    assert not ctx[d].is_polynomial()


def test_degree_of_linear(visitor, ctx):
    lin = dex.LinearExpression(
        children=[dex.Variable('x0', None, None)],
        coefficients=[-1.0],
    )
    visitor(lin, ctx)
    assert ctx[lin].degree == 1


def test_degree_of_pow_base_not_polynomial(visitor, ctx):
    base = PlaceholderExpression()
    expo = PlaceholderExpression()
    p = dex.PowExpression([base, expo])
    ctx[base] = PolynomialDegree.not_polynomial()
    ctx[expo] = PolynomialDegree(2)
    visitor(p, ctx)
    assert not ctx[p].is_polynomial()


def test_degree_of_pow_exponent_not_polynomial(visitor, ctx):
    base = PlaceholderExpression()
    expo = PlaceholderExpression()
    p = dex.PowExpression([base, expo])
    ctx[base] = PolynomialDegree(2)
    ctx[expo] = PolynomialDegree.not_polynomial()
    visitor(p, ctx)
    assert not ctx[p].is_polynomial()


def test_degree_of_pow_exponent_not_constant(visitor, ctx):
    base = PlaceholderExpression()
    expo = PlaceholderExpression()
    p = dex.PowExpression([base, expo])
    ctx[base] = PolynomialDegree(2)
    ctx[expo] = PolynomialDegree(1)
    visitor(p, ctx)
    assert not ctx[p].is_polynomial()


def test_degree_of_pow_exponent_constant(visitor, ctx):
    base = PlaceholderExpression()
    expo = dex.Constant(2.0)
    p = dex.PowExpression([base, expo])
    ctx[base] = PolynomialDegree(2)
    visitor(expo, ctx)
    visitor(p, ctx)
    assert ctx[p].degree == 4


def test_degree_of_sum_of_polynomials(visitor, ctx):
    p0 = PlaceholderExpression()
    p1 = PlaceholderExpression()
    s = dex.SumExpression([p0, p1])
    ctx[p0] = PolynomialDegree(3)
    ctx[p1] = PolynomialDegree(2)
    visitor(s, ctx)
    assert ctx[s].degree == 3


def test_degree_of_sum_of_non_polynomials(visitor, ctx):
    p0 = PlaceholderExpression()
    p1 = PlaceholderExpression()
    s = dex.SumExpression([p0, p1])
    ctx[p0] = PolynomialDegree(3)
    ctx[p1] = PolynomialDegree.not_polynomial()
    visitor(s, ctx)
    assert not ctx[s].is_polynomial()


def test_degree_of_negation(visitor, ctx):
    p0 = PlaceholderExpression()
    n = dex.NegationExpression([p0])
    ctx[p0] = PolynomialDegree(1)
    visitor(n, ctx)
    assert ctx[n].degree == 1


def test_degree_of_unary_function(visitor, ctx):
    p0 = PlaceholderExpression()
    u = dex.CosExpression([p0])
    ctx[p0] = PolynomialDegree(1)
    visitor(u, ctx)
    assert not ctx[u].is_polynomial()

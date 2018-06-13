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
import itertools
from hypothesis import given, assume
import hypothesis.strategies as st
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    cvx_description_to_cvx,
    coefficients,
    reals,
    ctx,
)

import suspect.dag.expressions as dex
from suspect.math.arbitrary_precision import pi
from suspect.interval import Interval
from suspect.convexity.convexity import Convexity
from suspect.convexity.propagation import (
    ConvexityPropagationVisitor,
)


@pytest.fixture
def visitor():
    return ConvexityPropagationVisitor()


def test_variable_is_linear(visitor, ctx):
    var = dex.Variable('x0', None, None)
    visitor.visit(var, ctx)
    assert ctx.convexity[var].is_linear()


def test_constant_is_linear(visitor, ctx):
    const = dex.Constant(-1.0)
    visitor.visit(const, ctx)
    assert ctx.convexity[const].is_linear()


@pytest.fixture
def mock_constraint_visitor(visitor, ctx):
    def _f(children_cvx, lower_bound, upper_bound):
        f = PlaceholderExpression()
        ctx.convexity[f] = cvx_description_to_cvx(children_cvx)
        con = dex.Constraint(
            'c',
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            children=[f],
        )
        visitor.visit(con, ctx)
        return ctx.convexity[con]
    return _f


@pytest.mark.parametrize('children_cvx,lower_bound,upper_bound,expected', [
    # l <= g(x) <= u is always unknown except if g(x) is linear
    ('linear', -1.0, 1.0, Convexity.Linear),
    ('convex', -1.0, 1.0, Convexity.Unknown),
    ('concave', -1.0, 1.0, Convexity.Unknown),
    ('unknown', -1.0, 1.0, Convexity.Unknown),

    # g(x) <= u is the same as g(x)
    ('linear', None, 1.0, Convexity.Linear),
    ('convex', None, 1.0, Convexity.Convex),
    ('concave', None, 1.0, Convexity.Concave),
    ('unknown', None, 1.0, Convexity.Unknown),

    # l <= g(x) is the negation of g(x)
    ('linear', 1.0, None, Convexity.Linear),
    ('convex', 1.0, None, Convexity.Concave),
    ('concave', 1.0, None, Convexity.Convex),
    ('unknown', 1.0, None, Convexity.Unknown),
])
def test_constraint_convexity(mock_constraint_visitor, children_cvx,
                              lower_bound, upper_bound, expected):
    cvx = mock_constraint_visitor(children_cvx, lower_bound, upper_bound)
    assert cvx == expected


@pytest.fixture
def mock_objective_visitor(visitor, ctx):
    def _f(sense, children_cvx):
        f = PlaceholderExpression()
        ctx.convexity[f] = cvx_description_to_cvx(children_cvx)
        obj = dex.Objective('obj', sense, children=[f])
        visitor.visit(obj, ctx)
        return ctx.convexity[obj]
    return _f


class TestObjective(object):
    @pytest.mark.parametrize("children, expected", [
        ('convex', Convexity.Convex),
        ('concave', Convexity.Concave),
        ('linear', Convexity.Linear),
        ('unknown', Convexity.Unknown),
    ])
    def test_min(self, mock_objective_visitor, children, expected):
        mono = mock_objective_visitor(dex.Sense.MINIMIZE, children)
        assert mono == expected

    @pytest.mark.parametrize("children, expected", [
        ('convex', Convexity.Concave),
        ('concave', Convexity.Convex),
        ('linear', Convexity.Linear),
        ('unknown', Convexity.Unknown),
    ])
    def test_max(self, mock_objective_visitor, children, expected):
        mono = mock_objective_visitor(dex.Sense.MAXIMIZE, children)
        assert mono == expected


@pytest.fixture
def mock_product_visitor(visitor, ctx):
    def _f(cvx_f, bound_g):
        f = PlaceholderExpression()
        ctx.convexity[f] = cvx_description_to_cvx(cvx_f)
        ctx.monotonicity[f] = mono_description_to_mono('unknown')
        g = PlaceholderExpression()
        ctx.convexity[g] = cvx_description_to_cvx('linear')  # g is constant
        ctx.monotonicity[g] = mono_description_to_mono('constant')
        ctx.bound[g] = bound_description_to_bound(bound_g)
        prod_fg = dex.ProductExpression([f, g])
        prod_gf = dex.ProductExpression([g, f])
        visitor.visit(prod_fg, ctx)
        visitor.visit(prod_gf, ctx)
        # product is commutative
        assert ctx.convexity[prod_fg] == ctx.convexity[prod_gf]
        return ctx.convexity[prod_fg]
    return _f


class TestProduct(object):
    @pytest.mark.parametrize('cvx_f,bound_g,expected', [
        ('convex', 'nonnegative', Convexity.Convex),
        ('convex', 'nonpositive', Convexity.Concave),
        ('concave', 'nonnegative', Convexity.Concave),
        ('concave', 'nonpositive', Convexity.Convex),
        ('linear', 'nonnegative', Convexity.Linear),
        ('linear', 'nonpositive', Convexity.Linear),
        ])
    def test_product_with_constant(self, mock_product_visitor, cvx_f, bound_g, expected):
        cvx = mock_product_visitor(cvx_f, bound_g)
        assert cvx == expected


@pytest.fixture
def mock_division_visitor(visitor, ctx):
    def _f(cvx_f, mono_f, bound_f, cvx_g, mono_g, bound_g):
        f = PlaceholderExpression()
        ctx.convexity[f] = cvx_description_to_cvx(cvx_f)
        ctx.monotonicity[f] = mono_description_to_mono(mono_f)
        ctx.bound[f] = bound_description_to_bound(bound_f)

        g = PlaceholderExpression()
        ctx.convexity[g] = cvx_description_to_cvx(cvx_g)
        ctx.monotonicity[g] = mono_description_to_mono(mono_g)
        ctx.bound[g] = bound_description_to_bound(bound_g)

        div = dex.DivisionExpression([f, g])
        visitor.visit(div, ctx)
        return ctx.convexity[div]
    return _f


class TestDivision(object):
    @pytest.mark.parametrize('cvx_f,mono_f,bound_f,cvx_g,mono_g,bound_g,expected', [
        ('convex', 'unknown', 'unbounded', 'linear', 'constant', 'positive', Convexity.Convex),
        ('convex', 'unknown', 'unbounded', 'linear', 'constant', 'nonnegative', Convexity.Unknown),

        ('concave', 'unknown', 'unbounded', 'linear', 'constant', 'negative', Convexity.Convex),
        ('concave', 'unknown', 'unbounded', 'linear', 'constant', 'nonpositive', Convexity.Unknown),

        ('convex', 'unknown', 'unbounded', 'linear', 'constant', 'negative', Convexity.Concave),
        ('convex', 'unknown', 'unbounded', 'linear', 'constant', 'nonpositive', Convexity.Unknown),

        ('concave', 'unknown', 'unbounded', 'linear', 'constant', 'positive', Convexity.Concave),
        ('concave', 'unknown', 'unbounded', 'linear', 'constant', 'nonnegative', Convexity.Unknown),
    ])
    def test_constant_denominator(self, mock_division_visitor, cvx_f, mono_f,
                                  bound_f, cvx_g, mono_g, bound_g, expected):
        cvx = mock_division_visitor(cvx_f, mono_f, bound_f,
                                    cvx_g, mono_g, bound_g)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_f,mono_f,bound_f,cvx_g,mono_g,bound_g,expected', [
        ('linear', 'constant', 'nonnegative', 'concave', 'unknown', 'positive', Convexity.Convex),
        ('linear', 'constant', 'nonnegative', 'concave', 'unknown', 'nonnegative', Convexity.Unknown),

        ('linear', 'constant', 'nonpositive', 'convex', 'unknown', 'negative', Convexity.Convex),
        ('linear', 'constant', 'nonpositive', 'convex', 'unknown', 'nonpositive', Convexity.Unknown),

        ('linear', 'constant', 'nonnegative', 'convex', 'unknown', 'negative', Convexity.Concave),
        ('linear', 'constant', 'nonnegative', 'convex', 'unknown', 'nonpositive', Convexity.Unknown),

        ('linear', 'constant', 'nonpositive', 'concave', 'unknown', 'positive', Convexity.Concave),
        ('linear', 'constant', 'nonpositive', 'concave', 'unknown', 'nonnegative', Convexity.Unknown),
    ])
    def test_constant_numerator(self, mock_division_visitor, cvx_f, mono_f,
                                bound_f, cvx_g, mono_g, bound_g, expected):
        cvx = mock_division_visitor(cvx_f, mono_f, bound_f,
                                    cvx_g, mono_g, bound_g)
        assert cvx == expected


@pytest.fixture
def mock_linear_visitor(visitor, ctx):
    def _f(terms):
        coefs = [c for c, _ in terms]
        children = [PlaceholderExpression() for _ in terms]
        cvxs = [c for _, c in terms]
        for child, cvx in zip(children, cvxs):
            ctx.convexity[child] = cvx_description_to_cvx(cvx)
        linear = dex.LinearExpression(coefs, children)
        visitor.visit(linear, ctx)
        return ctx.convexity[linear]
    return _f


@st.composite
def convex_terms(draw):
    coef = draw(coefficients())
    if coef > 0:
        return (coef, 'convex')
    else:
        return (coef, 'concave')


@st.composite
def concave_terms(draw):
    coef = draw(coefficients())
    if coef < 0:
        return (coef, 'convex')
    else:
        return (coef, 'concave')


@st.composite
def linear_terms(draw):
    coef = draw(coefficients())
    return (coef, 'linear')


class TestLinear(object):
    @given(
        st.lists(convex_terms(), min_size=1),
        st.lists(linear_terms()),
    )
    def test_convex(self, mock_linear_visitor, a, b):
        cvx = mock_linear_visitor(a + b)
        assert cvx.is_convex()

    @given(
        st.lists(concave_terms(), min_size=1),
        st.lists(linear_terms()),
    )
    def test_concave(self, mock_linear_visitor, a, b):
        cvx = mock_linear_visitor(a + b)
        assert cvx.is_concave()

    @given(
        st.lists(linear_terms(), min_size=1),
    )
    def test_concave(self, mock_linear_visitor, a):
        cvx = mock_linear_visitor(a)
        assert cvx.is_linear()

    @given(
        st.lists(convex_terms(), min_size=1),
        st.lists(concave_terms(), min_size=1),
    )
    def test_unknown(self, mock_linear_visitor, a, b):
        terms = a + b
        for c, _ in terms:
            assume(c != 0.0)
        cvx = mock_linear_visitor(terms)
        assert cvx.is_unknown()


@pytest.fixture
def mock_sum_visitor(visitor, ctx):
    def _f(cvxs):
        children = [PlaceholderExpression() for _ in cvxs]
        for child, cvx in zip(children, cvxs):
            ctx.convexity[child] = cvx_description_to_cvx(cvx)
        sum_ = dex.SumExpression(children)
        visitor.visit(sum_, ctx)
        return ctx.convexity[sum_]
    return _f


class TestSum(object):
    @given(
        st.lists(st.just('convex'), min_size=1),
        st.lists(st.just('linear')),
    )
    def test_convex(self, mock_sum_visitor, a, b):
        cvx = mock_sum_visitor(a + b)
        assert cvx.is_convex()

    @given(
        st.lists(st.just('concave'), min_size=1),
        st.lists(st.just('linear')),
    )
    def test_concave(self, mock_sum_visitor, a, b):
        cvx = mock_sum_visitor(a + b)
        assert cvx.is_concave()

    @given(
        st.lists(st.just('linear'), min_size=1),
    )
    def test_linear(self, mock_sum_visitor, a):
        cvx = mock_sum_visitor(a)
        assert cvx.is_linear()

    @given(
        st.lists(st.just('concave'), min_size=1),
        st.lists(st.just('convex'), min_size=1),
    )
    def test_concave(self, mock_sum_visitor, a, b):
        cvx = mock_sum_visitor(a + b)
        assert cvx.is_unknown()


@pytest.fixture
def mock_unary_function_visitor(visitor, ctx):
    def _f(func, cvx_g, mono_g, bound_g):
        g = PlaceholderExpression()
        ctx.convexity[g] = cvx_description_to_cvx(cvx_g)
        ctx.monotonicity[g] = mono_description_to_mono(mono_g)
        ctx.bound[g] = bound_description_to_bound(bound_g)
        expr = func([g])
        visitor.visit(expr, ctx)
        return ctx.convexity[expr]
    return _f


class TestAbs(object):
    @pytest.mark.parametrize('mono_g,bound_g',
                             itertools.product(['nondecreasing', 'nonincreasing', 'constant'],
                                               ['nonpositive', 'nonnegative', 'unbounded']))
    def test_linear_child(self, mock_unary_function_visitor, mono_g, bound_g):
        cvx = mock_unary_function_visitor(dex.AbsExpression, 'linear', mono_g, bound_g)
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('mono_g,bound_g,expected', [
        ('unknown', 'nonnegative', Convexity.Convex),
        ('unknown', 'nonpositive', Convexity.Concave),
    ])
    def test_convex_child(self, mock_unary_function_visitor, mono_g, bound_g, expected):
        cvx = mock_unary_function_visitor(dex.AbsExpression, 'convex', mono_g, bound_g)
        assert cvx == expected

    @pytest.mark.parametrize('mono_g,bound_g,expected', [
        ('unknown', 'nonnegative', Convexity.Concave),
        ('unknown', 'nonpositive', Convexity.Convex),
    ])
    def test_concave_child(self, mock_unary_function_visitor, mono_g, bound_g, expected):
        cvx = mock_unary_function_visitor(dex.AbsExpression, 'concave', mono_g, bound_g)
        assert cvx == expected


class TestSqrt(object):
    @pytest.mark.parametrize(
        'cvx_g,mono_g',
        itertools.product(
            ['concave', 'linear'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown'],
        )
    )
    def test_concave_child(self, mock_unary_function_visitor, cvx_g, mono_g):
        cvx = mock_unary_function_visitor(dex.SqrtExpression, cvx_g, mono_g, 'nonnegative')
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize(
        'cvx_g,mono_g',
        itertools.product(
            ['convex', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown']
        ),
    )
    def test_non_concave_child(self, mock_unary_function_visitor, cvx_g, mono_g):
        cvx = mock_unary_function_visitor(dex.SqrtExpression, cvx_g, mono_g, 'nonnegative')
        assert cvx == Convexity.Unknown


class TestExp(object):
    @pytest.mark.parametrize(
        'cvx_g,mono_g,bound_g',
        itertools.product(
            ['convex', 'linear'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown'],
            ['nonnegative', 'nonpositive', 'unbounded'],
        )
    )
    def test_convex_child(self, mock_unary_function_visitor, cvx_g, mono_g, bound_g):
        cvx = mock_unary_function_visitor(dex.ExpExpression, cvx_g, mono_g, bound_g)
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize(
        'cvx_g,mono_g,bound_g',
        itertools.product(
            ['concave', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown'],
            ['nonnegative', 'nonpositive', 'unbounded'],
        )
    )
    def test_convex_child(self, mock_unary_function_visitor, cvx_g, mono_g, bound_g):
        cvx = mock_unary_function_visitor(dex.ExpExpression, cvx_g, mono_g, bound_g)
        assert cvx == Convexity.Unknown


class TestLog(object):
    @pytest.mark.parametrize(
        'cvx_g,mono_g',
        itertools.product(
            ['concave', 'linear'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown'],
        )
    )
    def test_concave_child(self, mock_unary_function_visitor, cvx_g, mono_g):
        cvx = mock_unary_function_visitor(dex.LogExpression, cvx_g, mono_g, 'nonnegative')
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize(
        'cvx_g,mono_g',
        itertools.product(
            ['convex', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown']
        ),
    )
    def test_non_concave_child(self, mock_unary_function_visitor, cvx_g, mono_g):
        cvx = mock_unary_function_visitor(dex.LogExpression, cvx_g, mono_g, 'nonnegative')
        assert cvx == Convexity.Unknown


class TestSin(object):
    def test_bound_size_too_big(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Interval(-2, 2))
        assert cvx == Convexity.Unknown

    def test_bound_opposite_sign(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Interval(-0.1, 0.1))
        assert cvx == Convexity.Unknown

    def test_positive_sin_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Interval(pi/2-0.1, pi/2+0.1))
        assert cvx == Convexity.Concave

    def test_positive_sin_convex_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'convex', 'unknown', Interval(pi/2-0.1, pi/2+0.1))
        assert cvx == Convexity.Unknown

    def test_positive_sin_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'convex', 'unknown', Interval(pi/2+0.1, pi-0.1))
        assert cvx == Convexity.Concave

    def test_positive_sin_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'concave', 'unknown', Interval(0.1, pi/2-0.1))
        assert cvx == Convexity.Concave

    def test_negative_sin_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Interval(1.5*pi-0.1, 1.5*pi+0.1))
        assert cvx == Convexity.Convex

    def test_negative_sin_concave_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'concave', 'unknown', Interval(1.5*pi-0.1, 1.5*pi+0.1))
        assert cvx == Convexity.Unknown

    def test_negative_sin_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'concave', 'unknown', Interval(pi+0.1, 1.5*pi-0.1))
        assert cvx == Convexity.Convex

    def test_negative_sin_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'convex', 'unknown', Interval(1.5*pi, 2*pi))
        assert cvx == Convexity.Convex


class TestCos(object):
    def test_bound_size_too_big(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Interval(-2, 2))
        assert cvx == Convexity.Unknown

    def test_bound_opposite_sign(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Interval(pi/2-0.1, pi/2+0.1))
        assert cvx == Convexity.Unknown

    def test_positive_cos_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Interval(0, 0.5*pi))
        assert cvx == Convexity.Concave

    def test_positive_cos_convex_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'convex', 'unknown', Interval(0, 0.5*pi))
        assert cvx == Convexity.Unknown

    def test_positive_cos_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'convex', 'unknown', Interval(1.5*pi, 2*pi))
        assert cvx == Convexity.Concave

    def test_positive_cos_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'concave', 'unknown', Interval(0.0, 0.5*pi))
        assert cvx == Convexity.Concave

    def test_negative_cos_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Interval(0.6*pi, 1.4*pi))
        assert cvx == Convexity.Convex

    def test_negative_cos_concave_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'concave', 'unknown', Interval(0.6*pi, 0.9*pi))
        assert cvx == Convexity.Unknown

    def test_negative_cos_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'concave', 'unknown', Interval(1.1*pi, 1.4*pi))
        assert cvx == Convexity.Convex

    def test_negative_cos_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'convex', 'unknown', Interval(0.6*pi, 0.9*pi))
        assert cvx == Convexity.Convex



class TestTan(object):
    def test_bound_size_too_big(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'linear', 'constant', Interval(-2, 2))
        assert cvx == Convexity.Unknown

    def test_bound_opposite_sign(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'linear', 'constant', Interval(-0.1, 0.1))
        assert cvx == Convexity.Unknown

    def test_positive_tan_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'convex', 'unknown', Interval(pi, 1.5*pi))
        assert cvx == Convexity.Convex

    def test_positive_tan_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'concave', 'unknown', Interval(pi, 1.5*pi))
        assert cvx == Convexity.Unknown

    def test_negative_tan_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'convex', 'unknown', Interval(-1.5*pi, -pi))
        assert cvx == Convexity.Unknown

    def test_negative_tan_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'concave', 'unknown', Interval(-0.49*pi, -0.1))
        assert cvx == Convexity.Concave


class TestAsin(object):
    def test_is_concave(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'concave', 'unknown', Interval(-1, 0))
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize('bound_g', [Interval(-1, 0.1), Interval(-1.1, 0.0), Interval(None, None)])
    def test_is_not_concave(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'concave', 'unknown', bound_g)
        assert cvx == Convexity.Unknown

    def test_is_convex(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'convex', 'unknown', Interval(0, 1))
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('bound_g', [Interval(-0.1, 1), Interval(0.0, 1.1), Interval(None, None)])
    def test_is_not_convex(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'convex', 'unknown', bound_g)
        assert cvx == Convexity.Unknown


class TestAcos(object):
    def test_is_concave(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'convex', 'unknown', Interval(0, 1))
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize('bound_g', [Interval(-0.1, 1), Interval(0.0, 1.1), Interval(None, None)])
    def test_is_not_concave(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'convex', 'unknown', bound_g)
        assert cvx == Convexity.Unknown

    def test_is_convex(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'concave', 'unknown', Interval(-1, 0))
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('bound_g', [Interval(-1, 0.1), Interval(-1.1, 0.0), Interval(None, None)])
    def test_is_not_convex(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'concave', 'unknown', bound_g)
        assert cvx == Convexity.Unknown


class TestAtan(object):
    def test_is_concave(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AtanExpression, 'concave', 'unknown', 'nonnegative')
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize('bound_g', ['nonpositive', 'unbounded'])
    def test_is_not_concave(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AtanExpression, 'concave', 'unknown', bound_g)
        assert cvx == Convexity.Unknown

    def test_is_convex(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AtanExpression, 'convex', 'unknown', 'nonpositive')
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('bound_g', ['nonnegative', 'unbounded'])
    def test_is_not_convex(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AtanExpression, 'convex', 'unknown', bound_g)
        assert cvx == Convexity.Unknown


@pytest.fixture
def mock_pow_constant_base_visitor(visitor, ctx):
    def _f(bound_b, cvx_e, mono_e, bound_e):
        b = dex.Constant(bound_b)
        ctx.monotonicity[b] = mono_description_to_mono('constant')
        ctx.bound[b] = bound_description_to_bound(bound_b)
        e = PlaceholderExpression()
        ctx.convexity[e] = cvx_description_to_cvx(cvx_e)
        ctx.monotonicity[e] = mono_description_to_mono(mono_e)
        ctx.bound[e] = bound_description_to_bound(bound_e)
        p = dex.PowExpression([b, e])
        visitor.visit(p, ctx)
        return ctx.convexity[p]
    return _f


class TestPowConstantBase(object):
    @pytest.mark.parametrize(
        'cvx_e,mono_e,bound_e',
        itertools.product(
            ['convex', 'concave', 'linear', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'unknown'],
            ['nonpositive', 'nonnegative', 'unbounded']
        )
    )
    @given(base=reals(max_value=-0.01, allow_infinity=False))
    def test_negative_base(self, mock_pow_constant_base_visitor, base,
                           cvx_e, mono_e, bound_e):
        cvx = mock_pow_constant_base_visitor(base, cvx_e, mono_e, bound_e)
        assert cvx == Convexity.Unknown

    @pytest.mark.parametrize('cvx_e,mono_e,bound_e',
        itertools.product(
            ['convex', 'concave', 'linear', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'unknown'],
            ['nonpositive', 'nonnegative', 'unbounded']
        )
    )
    @given(base=reals(min_value=0.001, max_value=0.999))
    def test_base_between_0_and_1(self, mock_pow_constant_base_visitor, base,
                                  cvx_e, mono_e, bound_e):
        if cvx_e == 'concave' or cvx_e == 'linear':
            expected = Convexity.Convex
        else:
            expected = Convexity.Unknown
        cvx = mock_pow_constant_base_visitor(base, cvx_e, mono_e, bound_e)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_e,mono_e,bound_e',
        itertools.product(
            ['convex', 'concave', 'linear', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'unknown'],
            ['nonpositive', 'nonnegative', 'unbounded']
        )
    )
    @given(base=reals(min_value=1, allow_infinity=False))
    def test_base_gt_1(self, mock_pow_constant_base_visitor, base,
                       cvx_e, mono_e, bound_e):
        if cvx_e == 'convex' or cvx_e == 'linear':
            expected = Convexity.Convex
        else:
            expected = Convexity.Unknown
        cvx = mock_pow_constant_base_visitor(base, cvx_e, mono_e, bound_e)
        assert cvx == expected


@pytest.fixture
def mock_pow_constant_exponent_visitor(visitor, ctx):
    def _f(cvx_b, mono_b, bound_b, expo):
        expo = dex.Constant(expo)
        ctx.monotonicity[expo] = mono_description_to_mono('constant')
        ctx.bound[expo] = Interval(expo.value, expo.value)
        visitor.visit(expo, ctx)

        base = PlaceholderExpression()
        ctx.convexity[base] = cvx_description_to_cvx(cvx_b)
        ctx.monotonicity[base] = mono_description_to_mono(mono_b)
        ctx.bound[base] = bound_description_to_bound(bound_b)
        p = dex.PowExpression([base, expo])
        visitor.visit(p, ctx)
        return ctx.convexity[p]
    return _f


class TestPowConstantExponent(object):
    @pytest.mark.parametrize(
        'cvx_b,mono_b,bound_b',
        itertools.product(
            ['convex', 'concave', 'linear', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown'],
            ['nonpositive', 'nonnegative', 'unbounded']
        )
    )
    def test_exponent_equals_0(self, mock_pow_constant_exponent_visitor, cvx_b, mono_b, bound_b):
        cvx = mock_pow_constant_exponent_visitor(cvx_b, mono_b, bound_b, 0.0)
        assert cvx == Convexity.Linear

    @pytest.mark.parametrize(
        'cvx_b,mono_b,bound_b',
        itertools.product(
            ['convex', 'concave', 'linear', 'unknown'],
            ['nondecreasing', 'nonincreasing', 'constant', 'unknown'],
            ['nonpositive', 'nonnegative', 'unbounded']
        )
    )
    def test_exponent_equals_1(self, mock_pow_constant_exponent_visitor, cvx_b, mono_b, bound_b):
        cvx = mock_pow_constant_exponent_visitor(cvx_b, mono_b, bound_b, 1.0)
        assert cvx == cvx_description_to_cvx(cvx_b)

    @pytest.mark.parametrize('cvx_b,mono_b,bound_b,expected', [
        ('linear', 'nondecreasing', 'unbounded', Convexity.Convex),

        ('convex', 'unknown', 'nonnegative', Convexity.Convex),
        ('convex', 'unknown', 'nonpositive', Convexity.Unknown),

        ('concave', 'unknown', 'nonnegative', Convexity.Unknown),
        ('concave', 'unknown', 'nonpositive', Convexity.Convex),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_even_integer(self, mock_pow_constant_exponent_visitor, expo,
                                   cvx_b, mono_b, bound_b, expected):
        cvx = mock_pow_constant_exponent_visitor(cvx_b, mono_b, bound_b, 2*expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_b,mono_b,bound_b,expected', [
        ('convex', 'unknown', 'nonpositive', Convexity.Convex),
        ('convex', 'unknown', 'nonnegative', Convexity.Concave),

        ('concave', 'unknown', 'nonnegative', Convexity.Convex),
        ('concave', 'unknown', 'nonpositive', Convexity.Concave),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_even_integer(self, mock_pow_constant_exponent_visitor, expo,
                                   cvx_b, mono_b, bound_b, expected):
        cvx = mock_pow_constant_exponent_visitor(cvx_b, mono_b, bound_b, -2*expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_b,mono_b,bound_b,expected', [
        ('convex', 'unknown', 'nonnegative', Convexity.Convex),
        ('convex', 'unknown', 'nonpositive', Convexity.Unknown),
        ('concave', 'unknown', 'nonpositive', Convexity.Concave),
        ('concave', 'unknown', 'nonnegative', Convexity.Unknown),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_odd_integer(self, mock_pow_constant_exponent_visitor, expo,
                                  cvx_b, mono_b, bound_b, expected):
        cvx = mock_pow_constant_exponent_visitor(cvx_b, mono_b, bound_b, 2*expo+1)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_b,mono_b,bound_b,expected', [
        ('concave', 'unknown', 'nonnegative', Convexity.Convex),
        ('concave', 'unknown', 'nonpositive', Convexity.Unknown),
        ('convex', 'unknown', 'nonpositive', Convexity.Concave),
        ('convex', 'unknown', 'nonnegative', Convexity.Unknown),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_odd_integer(self, mock_pow_constant_exponent_visitor, expo,
                                  cvx_b, mono_b, bound_b, expected):
        cvx = mock_pow_constant_exponent_visitor(cvx_b, mono_b, bound_b, -2*expo+1)
        assert cvx == expected

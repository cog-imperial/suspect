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
)

import suspect.dag.expressions as dex
from suspect.math.arbitrary_precision import pi
from suspect.bound import ArbitraryPrecisionBound as Bound
from suspect.convexity.convexity import Convexity
from suspect.convexity.propagation import (
    ConvexityPropagationVisitor,
)


class MockConvexityPropagationVisitor(ConvexityPropagationVisitor):
    def __init__(self):
        super().__init__({}, {})
        self._cvx = {}

    def add_bound(self, expr, bound_str):
        self._bounds[id(expr)] = bound_description_to_bound(bound_str)

    def add_mono(self, expr, mono_str):
        self._mono[id(expr)] = mono_description_to_mono(mono_str)

    def add_cvx(self, cvx_str):
        node = PlaceholderExpression()
        self._cvx[id(node)] = cvx_description_to_cvx(cvx_str)
        return node


def test_variable_is_linear():
    v = MockConvexityPropagationVisitor()
    var = dex.Variable('x0', None, None)
    v(var)
    assert v.get(var).is_linear()


def test_constant_is_linear():
    v = MockConvexityPropagationVisitor()
    const = dex.Constant(-1.0)
    v(const)
    assert v.get(const).is_linear()


@pytest.fixture
def mock_constraint_visitor():
    def _f(children_cvx, lower_bound, upper_bound):
        v = MockConvexityPropagationVisitor()
        f = v.add_cvx(children_cvx)
        con = dex.Constraint('c',
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            children=[f],
        )
        v(con)
        return v.get(con)
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
def mock_objective_visitor():
    def _f(sense, children_cvx):
        v = MockConvexityPropagationVisitor()
        f = v.add_cvx(children_cvx)
        obj = dex.Objective('obj', sense, children=[f])
        v(obj)
        return v.get(obj)
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
def mock_product_visitor():
    def _f(cvx_f, bound_g):
        v = MockConvexityPropagationVisitor()
        f = v.add_cvx(cvx_f)
        v.add_mono(f, 'unknown')
        g = v.add_cvx('linear')  # g is constant
        v.add_bound(g, bound_g)
        v.add_mono(g, 'constant')
        prod_fg = dex.ProductExpression([f, g])
        prod_gf = dex.ProductExpression([g, f])
        v(prod_fg)
        v(prod_gf)
        # product is commutative
        assert v.get(prod_fg) == v.get(prod_gf)
        return v.get(prod_fg)
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
def mock_division_visitor():
    def _f(cvx_f, mono_f, bound_f, cvx_g, mono_g, bound_g):
        v = MockConvexityPropagationVisitor()
        f = v.add_cvx(cvx_f)
        v.add_mono(f, mono_f)
        v.add_bound(f, bound_f)
        g = v.add_cvx(cvx_g)
        v.add_mono(g, mono_g)
        v.add_bound(g, bound_g)
        div = dex.DivisionExpression([f, g])
        v(div)
        return v.get(div)
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
def mock_linear_visitor():
    def _f(terms):
        v = MockConvexityPropagationVisitor()
        coefs = [c for c, _ in terms]
        children = [v.add_cvx(x) for _, x in terms]
        linear = dex.LinearExpression(coefs, children)
        v(linear)
        return v.get(linear)
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
def mock_sum_visitor():
    def _f(cvxs):
        v = MockConvexityPropagationVisitor()
        children = [v.add_cvx(x) for x in cvxs]
        sum_ = dex.SumExpression(children)
        v(sum_)
        return v.get(sum_)
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
def mock_abs_visitor():
    def _f(cvx_g, mono_g, bound_g):
        v = MockConvexityPropagationVisitor()
        g = v.add_cvx(cvx_g)
        v.add_mono(g, mono_g)
        v.add_bound(g, bound_g)
        abs_ = dex.AbsExpression([g])
        v(abs_)
        return v.get(abs_)
    return _f


class TestAbs(object):
    @pytest.mark.parametrize('mono_g,bound_g',
                             itertools.product(['nondecreasing', 'nonincreasing', 'constant'],
                                               ['nonpositive', 'nonnegative', 'unbounded']))
    def test_linear_child(self, mock_abs_visitor, mono_g, bound_g):
        cvx = mock_abs_visitor('linear', mono_g, bound_g)
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('mono_g,bound_g,expected', [
        ('unknown', 'nonnegative', Convexity.Convex),
        ('unknown', 'nonpositive', Convexity.Concave),
    ])
    def test_convex_child(self, mock_abs_visitor, mono_g, bound_g, expected):
        cvx = mock_abs_visitor('convex', mono_g, bound_g)
        assert cvx == expected

    @pytest.mark.parametrize('mono_g,bound_g,expected', [
        ('unknown', 'nonnegative', Convexity.Concave),
        ('unknown', 'nonpositive', Convexity.Convex),
    ])
    def test_concave_child(self, mock_abs_visitor, mono_g, bound_g, expected):
        cvx = mock_abs_visitor('concave', mono_g, bound_g)
        assert cvx == expected



@pytest.fixture
def mock_unary_function_visitor():
    def _f(func, cvx_g, mono_g, bound_g):
        v = MockConvexityPropagationVisitor()
        g = v.add_cvx(cvx_g)
        v.add_mono(g, mono_g)
        v.add_bound(g, bound_g)
        expr = func([g])
        v(expr)
        return v.get(expr)
    return _f


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
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Bound(-2, 2))
        assert cvx == Convexity.Unknown

    def test_bound_opposite_sign(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Bound(-0.1, 0.1))
        assert cvx == Convexity.Unknown

    def test_positive_sin_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Bound(pi/2-0.1, pi/2+0.1))
        assert cvx == Convexity.Concave

    def test_positive_sin_convex_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'convex', 'unknown', Bound(pi/2-0.1, pi/2+0.1))
        assert cvx == Convexity.Unknown

    def test_positive_sin_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'convex', 'unknown', Bound(pi/2+0.1, pi-0.1))
        assert cvx == Convexity.Concave

    def test_positive_sin_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'concave', 'unknown', Bound(0.1, pi/2-0.1))
        assert cvx == Convexity.Concave

    def test_negative_sin_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'linear', 'constant', Bound(1.5*pi-0.1, 1.5*pi+0.1))
        assert cvx == Convexity.Convex

    def test_negative_sin_concave_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'concave', 'unknown', Bound(1.5*pi-0.1, 1.5*pi+0.1))
        assert cvx == Convexity.Unknown

    def test_negative_sin_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'concave', 'unknown', Bound(pi+0.1, 1.5*pi-0.1))
        assert cvx == Convexity.Convex

    def test_negative_sin_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.SinExpression, 'convex', 'unknown', Bound(1.5*pi, 2*pi))
        assert cvx == Convexity.Convex


class TestCos(object):
    def test_bound_size_too_big(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Bound(-2, 2))
        assert cvx == Convexity.Unknown

    def test_bound_opposite_sign(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Bound(pi/2-0.1, pi/2+0.1))
        assert cvx == Convexity.Unknown

    def test_positive_cos_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Bound(0, 0.5*pi))
        assert cvx == Convexity.Concave

    def test_positive_cos_convex_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'convex', 'unknown', Bound(0, 0.5*pi))
        assert cvx == Convexity.Unknown

    def test_positive_cos_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'convex', 'unknown', Bound(1.5*pi, 2*pi))
        assert cvx == Convexity.Concave

    def test_positive_cos_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'concave', 'unknown', Bound(0.0, 0.5*pi))
        assert cvx == Convexity.Concave

    def test_negative_cos_linear_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'linear', 'constant', Bound(0.6*pi, 1.4*pi))
        assert cvx == Convexity.Convex

    def test_negative_cos_concave_child_but_wrong_interval(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'concave', 'unknown', Bound(0.6*pi, 0.9*pi))
        assert cvx == Convexity.Unknown

    def test_negative_cos_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'concave', 'unknown', Bound(1.1*pi, 1.4*pi))
        assert cvx == Convexity.Convex

    def test_negative_cos_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.CosExpression, 'convex', 'unknown', Bound(0.6*pi, 0.9*pi))
        assert cvx == Convexity.Convex



class TestTan(object):
    def test_bound_size_too_big(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'linear', 'constant', Bound(-2, 2))
        assert cvx == Convexity.Unknown

    def test_bound_opposite_sign(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'linear', 'constant', Bound(-0.1, 0.1))
        assert cvx == Convexity.Unknown

    def test_positive_tan_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'convex', 'unknown', Bound(pi, 1.5*pi))
        assert cvx == Convexity.Convex

    def test_positive_tan_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'concave', 'unknown', Bound(pi, 1.5*pi))
        assert cvx == Convexity.Unknown

    def test_negative_tan_convex_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'convex', 'unknown', Bound(-1.5*pi, -pi))
        assert cvx == Convexity.Unknown

    def test_negative_tan_concave_child(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.TanExpression, 'concave', 'unknown', Bound(-0.49*pi, -0.1))
        assert cvx == Convexity.Concave


class TestAsin(object):
    def test_is_concave(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'concave', 'unknown', Bound(-1, 0))
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize('bound_g', [Bound(-1, 0.1), Bound(-1.1, 0.0), Bound(None, None)])
    def test_is_not_concave(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'concave', 'unknown', bound_g)
        assert cvx == Convexity.Unknown

    def test_is_convex(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'convex', 'unknown', Bound(0, 1))
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('bound_g', [Bound(-0.1, 1), Bound(0.0, 1.1), Bound(None, None)])
    def test_is_not_convex(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AsinExpression, 'convex', 'unknown', bound_g)
        assert cvx == Convexity.Unknown


class TestAcos(object):
    def test_is_concave(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'convex', 'unknown', Bound(0, 1))
        assert cvx == Convexity.Concave

    @pytest.mark.parametrize('bound_g', [Bound(-0.1, 1), Bound(0.0, 1.1), Bound(None, None)])
    def test_is_not_concave(self, mock_unary_function_visitor, bound_g):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'convex', 'unknown', bound_g)
        assert cvx == Convexity.Unknown

    def test_is_convex(self, mock_unary_function_visitor):
        cvx = mock_unary_function_visitor(dex.AcosExpression, 'concave', 'unknown', Bound(-1, 0))
        assert cvx == Convexity.Convex

    @pytest.mark.parametrize('bound_g', [Bound(-1, 0.1), Bound(-1.1, 0.0), Bound(None, None)])
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
def mock_pow_constant_base_visitor():
    def _f(bound_b, cvx_e, mono_e, bound_e):
        v = MockConvexityPropagationVisitor()
        b = dex.Constant(bound_b)
        v.add_mono(b, 'constant')
        v.add_bound(b, bound_b)
        e = v.add_cvx(cvx_e)
        v.add_mono(e, mono_e)
        v.add_bound(e, bound_e)
        p = dex.PowExpression([b, e])
        v(p)
        return v.get(p)
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
def mock_pow_constant_exponent_visitor():
    def _f(cvx_b, mono_b, bound_b, expo):
        v = MockConvexityPropagationVisitor()
        expo = dex.Constant(expo)
        v(expo)
        v.add_mono(expo, 'constant')
        v.add_bound(expo, Bound(expo.value, expo.value))
        base = v.add_cvx(cvx_b)
        v.add_mono(base, mono_b)
        v.add_bound(base, bound_b)
        p = dex.PowExpression([base, expo])
        v(p)
        return v.get(p)
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

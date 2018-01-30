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
from hypothesis import given, assume
import hypothesis.strategies as st
import itertools
from tests.conftest import (
    PlaceholderExpression,
    bound_description_to_bound,
    mono_description_to_mono,
    coefficients,
    reals,
)
import suspect.dag.expressions as dex
from suspect.math.arbitrary_precision import pi
from suspect.bound import ArbitraryPrecisionBound as Bound
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.propagation import (
    MonotonicityPropagationVisitor,
)


class MockMonotonicityPropagationVisitor(MonotonicityPropagationVisitor):
    def __init__(self):
        super().__init__({})
        self._mono = {}

    def add_bound(self, expr, bound_str):
        self._bounds[id(expr)] = bound_description_to_bound(bound_str)

    def add_mono(self, mono_str):
        node = PlaceholderExpression()
        self._mono[id(node)] = mono_description_to_mono(mono_str)
        return node


def test_variable_is_increasing():
    v = MockMonotonicityPropagationVisitor()
    var = dex.Variable('x0', None, None)
    v(var)
    assert v.get(var).is_nondecreasing()


def test_constant_is_constant():
    v = MockMonotonicityPropagationVisitor()
    const = dex.Constant(2.0)
    v(const)
    assert v.get(const).is_constant()


class TestConstraint(object):
    def setup_method(self, _func):
        self.x0 = dex.Variable('x0', None, None)
        self.x1 = dex.NegationExpression([self.x0])
        self.const = dex.Constant(1.0)
        self.v = MockMonotonicityPropagationVisitor()
        self.v(self.x0)
        self.v(self.x1)
        self.v(self.const)

    def test_lower_upper_bound(self):
        c0 = dex.Constraint('c0', 0, 1, [self.x0])
        self.v(c0)
        assert self.v.get(c0).is_unknown()

        c1 = dex.Constraint('c1', 0, 1, [self.const])
        self.v(c1)
        assert self.v.get(c1).is_constant()

    def test_lower_bound_only(self):
        c0 = dex.Constraint('c0', 0, None, [self.x0])
        self.v(c0)
        assert self.v.get(c0).is_nonincreasing()

        c1 = dex.Constraint('c1', 0, None, [self.const])
        self.v(c1)
        assert self.v.get(c1).is_constant()

        c2 = dex.Constraint('c2', 0, None, [self.x1])
        self.v(c2)
        assert self.v.get(c2).is_nondecreasing()

    def test_upper_bound_only(self):
        c0 = dex.Constraint('c0', None, 1, [self.x0])
        self.v(c0)
        assert self.v.get(c0).is_nondecreasing()

        c1 = dex.Constraint('c1', None, 1, [self.const])
        self.v(c1)
        assert self.v.get(c1).is_constant()

        c2 = dex.Constraint('c2', None, 1, [self.x1])
        self.v(c2)
        assert self.v.get(c2).is_nonincreasing()


@pytest.fixture
def mock_objective_visitor():
    def _f(sense, children_mono):
        v = MockMonotonicityPropagationVisitor()
        c = v.add_mono(children_mono)
        obj = dex.Objective('obj', sense, [c])
        v(obj)
        return v.get(obj)
    return _f


class TestObjective(object):
    @pytest.mark.parametrize("children, expected", [
        ('nondecreasing', Monotonicity.Nondecreasing),
        ('nonincreasing', Monotonicity.Nonincreasing),
        ('constant', Monotonicity.Constant),
        ('unknown', Monotonicity.Unknown),
    ])
    def test_min(self, mock_objective_visitor, children, expected):
        mono = mock_objective_visitor(dex.Sense.MINIMIZE, children)
        assert mono == expected

    @pytest.mark.parametrize("children, expected", [
        ('nondecreasing', Monotonicity.Nonincreasing),
        ('nonincreasing', Monotonicity.Nondecreasing),
        ('constant', Monotonicity.Constant),
        ('unknown', Monotonicity.Unknown),
    ])
    def test_max(self, mock_objective_visitor, children, expected):
        mono = mock_objective_visitor(dex.Sense.MAXIMIZE, children)
        assert mono == expected


@pytest.fixture
def mock_product_visitor():
    def _f(mono_f, bound_f, mono_g, bound_g):
        v = MockMonotonicityPropagationVisitor()
        f = v.add_mono(mono_f)
        v.add_bound(f, bound_f)
        g = v.add_mono(mono_g)
        v.add_bound(g, bound_g)
        prod_fg = dex.ProductExpression([f, g])
        prod_gf = dex.ProductExpression([g, f])
        v(prod_fg)
        v(prod_gf)
        # product is commutative
        assert v.get(prod_fg) == v.get(prod_gf)
        return v.get(prod_fg)
    return _f


@pytest.mark.parametrize("mono_f,bound_f,mono_g,bound_g,expected", [
    # mono_f           bound_f      mono_g      bound_g
    ('nondecreasing', 'unbounded', 'constant', 'nonnegative', Monotonicity.Nondecreasing),
    ('nonincreasing', 'unbounded', 'constant', 'nonpositive', Monotonicity.Nondecreasing),

    ('nondecreasing', 'unbounded', 'constant', 'nonpositive', Monotonicity.Nonincreasing),
    ('nonincreasing', 'unbounded', 'constant', 'nonnegative', Monotonicity.Nonincreasing),

    ('constant', 'nonnegative', 'nondecreasing', 'unbounded', Monotonicity.Nondecreasing),
    ('constant', 'nonpositive', 'nonincreasing', 'unbounded', Monotonicity.Nondecreasing),

    ('constant', 'unbounded', 'constant', 'unbounded', Monotonicity.Constant),

    ('constant', 'nonpositive', 'nondecreasing', 'unbounded', Monotonicity.Nonincreasing),
    ('constant', 'nonnegative', 'nonincreasing', 'unbounded', Monotonicity.Nonincreasing),

    ('nondecreasing', 'nonnegative', 'nondecreasing', 'nonnegative', Monotonicity.Nondecreasing),
    ('nondecreasing', 'nonpositive', 'nonincreasing', 'nonnegative', Monotonicity.Nondecreasing),
    ('nonincreasing', 'nonnegative', 'nondecreasing', 'nonpositive', Monotonicity.Nondecreasing),
    ('nonincreasing', 'nonpositive', 'nonincreasing', 'nonpositive', Monotonicity.Nondecreasing),

    ('nonincreasing', 'nonnegative', 'nonincreasing', 'nonnegative', Monotonicity.Nonincreasing),
    ('nonincreasing', 'nonpositive', 'nondecreasing', 'nonnegative', Monotonicity.Nonincreasing),
    ('nondecreasing', 'nonnegative', 'nonincreasing', 'nonpositive', Monotonicity.Nonincreasing),
    ('nondecreasing', 'nonpositive', 'nondecreasing', 'nonpositive', Monotonicity.Nonincreasing),
])
def test_product(mock_product_visitor, mono_f, bound_f, mono_g, bound_g, expected):
    mono = mock_product_visitor(mono_f, bound_f, mono_g, bound_g)
    assert mono == expected


@pytest.fixture
def mock_division_visitor():
    def _f(mono_f, bound_f, mono_g, bound_g):
        v = MockMonotonicityPropagationVisitor()
        f = v.add_mono(mono_f)
        v.add_bound(f, bound_f)
        g = v.add_mono(mono_g)
        v.add_bound(g, bound_g)
        div = dex.DivisionExpression([f, g])
        v(div)
        return v.get(div)
    return _f


@pytest.mark.parametrize("mono_f,bound_f,mono_g,bound_g,expected", [
    # mono_f           bound_f      mono_g      bound_g
    ('constant',         1.0,       'constant',    2.0,       Monotonicity.Constant),
    ('constant',         1.0,       'constant',    0.0,       Monotonicity.Unknown),

    ('nondecreasing', 'unbounded', 'constant', 'nonnegative', Monotonicity.Nondecreasing),
    ('nonincreasing', 'unbounded', 'constant', 'nonpositive', Monotonicity.Nondecreasing),

    ('nondecreasing', 'unbounded', 'constant', 'nonpositive', Monotonicity.Nonincreasing),
    ('nonincreasing', 'unbounded', 'constant', 'nonnegative', Monotonicity.Nonincreasing),

    ('nondecreasing', 'nonnegative', 'nonincreasing', 'nonnegative', Monotonicity.Nondecreasing),
    ('nondecreasing', 'nonpositive', 'nondecreasing', 'nonnegative', Monotonicity.Nondecreasing),
    ('nonincreasing', 'nonnegative', 'nonincreasing', 'nonpositive', Monotonicity.Nondecreasing),
    ('nonincreasing', 'nonpositive', 'nondecreasing', 'nonpositive', Monotonicity.Nondecreasing),

    ('nonincreasing', 'nonnegative', 'nondecreasing', 'nonnegative', Monotonicity.Nonincreasing),
    ('nonincreasing', 'nonpositive', 'nonincreasing', 'nonnegative', Monotonicity.Nonincreasing),
    ('nondecreasing', 'nonnegative', 'nondecreasing', 'nonpositive', Monotonicity.Nonincreasing),
    ('nondecreasing', 'nonpositive', 'nonincreasing', 'nonpositive', Monotonicity.Nonincreasing),

])
def test_division(mock_division_visitor, mono_f, bound_f, mono_g, bound_g, expected):
    mono = mock_division_visitor(mono_f, bound_f, mono_g, bound_g)
    assert mono == expected


@pytest.fixture
def mock_linear_visitor():
    def _f(terms):
        v = MockMonotonicityPropagationVisitor()
        coefs = [c for c, _ in terms]
        children = [v.add_mono(m) for _, m in terms]
        linear = dex.LinearExpression(coefs, children)
        v(linear)
        return v.get(linear)
    return _f


@st.composite
def nondecreasing_terms(draw):
    coef = draw(coefficients())
    if coef > 0:
        return (coef, 'nondecreasing')
    else:
        return (coef, 'nonincreasing')


@st.composite
def nonincreasing_terms(draw):
    coef = draw(coefficients())
    if coef < 0:
        return (coef, 'nondecreasing')
    else:
        return (coef, 'nonincreasing')


@st.composite
def constant_terms(draw):
    coef = draw(coefficients())
    return (coef, 'constant')


class TestLinear(object):
    @given(
        st.lists(
            st.one_of(nondecreasing_terms(), constant_terms()),
            min_size=1)
    )
    def test_nondecreasing(self, mock_linear_visitor, terms):
        mono = mock_linear_visitor(terms)
        assert mono.is_nondecreasing()

    @given(
        st.lists(
            st.one_of(nonincreasing_terms(), constant_terms()),
            min_size=1)
    )
    def test_nonincreasing(self, mock_linear_visitor, terms):
        mono = mock_linear_visitor(terms)
        assert mono.is_nonincreasing()

    @given(st.lists(constant_terms(), min_size=1))
    def test_constant(self, mock_linear_visitor, terms):
        mono = mock_linear_visitor(terms)
        assert mono.is_constant()

    @given(
        st.lists(nondecreasing_terms(), min_size=1),
        st.lists(nonincreasing_terms(), min_size=1),
    )
    def test_unknown(self, mock_linear_visitor, a, b):
        terms = a + b
        for c, _ in terms:
            assume(c != 0.0)
        mono = mock_linear_visitor(terms)
        assert mono.is_unknown()


@pytest.fixture
def mock_sum_visitor():
    def _f(terms):
        v = MockMonotonicityPropagationVisitor()
        children = [v.add_mono(m) for m in terms]
        sum_ = dex.SumExpression(children)
        v(sum_)
        return v.get(sum_)
    return _f


class TestSum(object):
    @given(
        st.lists(st.just('nondecreasing'), min_size=1),
        st.lists(st.one_of(st.just('nondecreasing'), st.just('constant'))),
    )
    def test_nondecreasing(self, mock_sum_visitor, a, b):
        mono = mock_sum_visitor(a + b)
        assert mono.is_nondecreasing()

    @given(
        st.lists(st.just('nonincreasing'), min_size=1),
        st.lists(st.one_of(st.just('nonincreasing'), st.just('constant'))),
    )
    def test_nonincreasing(self, mock_sum_visitor, a, b):
        mono = mock_sum_visitor(a + b)
        assert mono.is_nonincreasing()

    @given(st.lists(st.just('constant'), min_size=1))
    def test_constant(self, mock_sum_visitor, a):
        mono = mock_sum_visitor(a)
        assert mono.is_constant()

    @given(
        st.lists(st.just('nonincreasing'), min_size=1),
        st.lists(st.just('nondecreasing'), min_size=1),
        st.lists(st.just('constant')),
    )
    def test_unknown(self, mock_sum_visitor, a, b, c):
        mono = mock_sum_visitor(a + b + c)
        assert mono.is_unknown()


@pytest.fixture
def mock_abs_visitor():
    def _f(mono, bound):
        v = MockMonotonicityPropagationVisitor()
        g = v.add_mono(mono)
        v.add_bound(g, bound)
        abs_ = dex.AbsExpression([g])
        v(abs_)
        return v.get(abs_)
    return _f


@pytest.mark.parametrize('mono_g,bound_g,expected', [
    ('nondecreasing', 'nonnegative', Monotonicity.Nondecreasing),
    ('nonincreasing', 'nonnegative', Monotonicity.Nonincreasing),
    ('nonincreasing', 'nonpositive', Monotonicity.Nondecreasing),
    ('nondecreasing', 'nonpositive', Monotonicity.Nonincreasing),
    ('nondecreasing', 'unbounded', Monotonicity.Unknown),
])
def test_abs(mock_abs_visitor, mono_g, bound_g, expected):
    mono = mock_abs_visitor(mono_g, bound_g)
    assert mono == expected


@pytest.fixture
def mock_nondecreasing_visitor():
    def _f(mono, bound, func):
        v = MockMonotonicityPropagationVisitor()
        g = v.add_mono(mono)
        v.add_bound(g, bound)
        f = func([g])
        v(f)
        return v.get(f)
    return _f


@pytest.mark.parametrize('func', [
    dex.SqrtExpression, dex.ExpExpression, dex.LogExpression,
    dex.AsinExpression, dex.AtanExpression
])
@given(
    mono_arg=st.one_of(
        st.just('constant'), st.just('nondecreasing'),
        st.just('nonincreasing'), st.just('unknown')
    ),
    bound_arg=st.one_of(
        st.just('zero'), st.just('nonpositive'), st.just('nonnegative')
    )
)
def test_nondecreasing_function(mock_nondecreasing_visitor, mono_arg,
                                bound_arg, func):
    mono = mock_nondecreasing_visitor(mono_arg, bound_arg, func)
    expected = {
        'nondecreasing': Monotonicity.Nondecreasing,
        'nonincreasing': Monotonicity.Nonincreasing,
        'constant': Monotonicity.Constant,
        'unknown': Monotonicity.Unknown,
    }[mono_arg]
    assert mono == expected


@st.composite
def nonnegative_cos_bounds(draw):
    start = draw(st.floats(
        min_value=1.6*pi,
        max_value=1.9*pi,
        allow_nan=False,
        allow_infinity=False,
    ))
    end = draw(st.floats(
        min_value=0,
        max_value=1.9*pi-start,
        allow_nan=False,
        allow_infinity=False,
    ))
    mul = draw(st.integers(min_value=-100, max_value=100)) * 2 * pi
    b = Bound(start + mul, start + end + mul)
    assume(b.size() > 0)
    return b


@st.composite
def nonpositive_cos_bounds(draw):
    start = draw(st.floats(
        min_value=0.6*pi,
        max_value=1.5*pi,
        allow_nan=False,
        allow_infinity=False,
    ))
    end = draw(st.floats(
        min_value=0,
        max_value=1.5*pi-start,
        allow_nan=False,
        allow_infinity=False,
    ))
    mul = draw(st.integers(min_value=-10, max_value=10)) * 2 * pi
    b = Bound(start + mul, start + end + mul)
    assume(b.size() > 0)
    return b


@pytest.fixture
def mock_sin_visitor():
    def _f(mono, bound):
        v = MockMonotonicityPropagationVisitor()
        g = v.add_mono(mono)
        v.add_bound(g, bound)
        sin_ = dex.SinExpression([g])
        v(sin_)
        return v.get(sin_)
    return _f


class TestSin(object):
    @given(bound_g=nonnegative_cos_bounds())
    def test_nondecreasing_nonnegative_cos(self, mock_sin_visitor, bound_g):
        mono = mock_sin_visitor('nondecreasing', bound_g)
        assert mono.is_nondecreasing()

    @given(bound_g=nonnegative_cos_bounds())
    def test_nonincreasing_nonnegative_cos(self, mock_sin_visitor, bound_g):
        mono = mock_sin_visitor('nonincreasing', bound_g)
        assert mono.is_nonincreasing()

    @given(bound_g=nonpositive_cos_bounds())
    def test_decreasing_nonpositive_cos(self, mock_sin_visitor, bound_g):
        mono = mock_sin_visitor('nondecreasing', bound_g)
        assert mono.is_nonincreasing()

    @given(bound_g=nonpositive_cos_bounds())
    def test_nonincreasing_nonpositive_cos(self, mock_sin_visitor, bound_g):
        mono = mock_sin_visitor('nonincreasing', bound_g)
        assert mono.is_nondecreasing()


@st.composite
def nonnegative_sin_bounds(draw):
    start = draw(st.floats(
        min_value=0.1*pi,
        max_value=0.9*pi,
        allow_nan=False,
        allow_infinity=False,
    ))
    end = draw(st.floats(
        min_value=0,
        max_value=0.9*pi-start,
        allow_nan=False,
        allow_infinity=False,
    ))
    mul = draw(st.integers(min_value=-100, max_value=100)) * 2 * pi
    b = Bound(start + mul, start + end + mul)
    assume(b.size() > 0)
    return b


@st.composite
def nonpositive_sin_bounds(draw):
    start = draw(st.floats(
        min_value=1.1*pi,
        max_value=1.9*pi,
        allow_nan=False,
        allow_infinity=False,
    ))
    end = draw(st.floats(
        min_value=0,
        max_value=1.9*pi-start,
        allow_nan=False,
        allow_infinity=False,
    ))
    mul = draw(st.integers(min_value=-10, max_value=10)) * 2 * pi
    b = Bound(start + mul, start + end + mul)
    assume(b.size() > 0)
    return b


@pytest.fixture
def mock_cos_visitor():
    def _f(mono, bound):
        v = MockMonotonicityPropagationVisitor()
        g = v.add_mono(mono)
        v.add_bound(g, bound)
        cos_ = dex.CosExpression([g])
        v(cos_)
        return v.get(cos_)
    return _f


class TestCos(object):
    @given(bound_g=nonnegative_sin_bounds())
    def test_nonincreasing_nonnegative_sin(self, mock_cos_visitor, bound_g):
        mono = mock_cos_visitor('nonincreasing', bound_g)
        assert mono.is_nondecreasing()

    @given(bound_g=nonnegative_sin_bounds())
    def test_nondecreasing_nonnegative_sin(self, mock_cos_visitor, bound_g):
        mono = mock_cos_visitor('nondecreasing', bound_g)
        assert mono.is_nonincreasing()

    @given(bound_g=nonpositive_sin_bounds())
    def test_nondecreasing_nonpositive_sin(self, mock_cos_visitor, bound_g):
        mono = mock_cos_visitor('nondecreasing', bound_g)
        assert mono.is_nondecreasing()

    @given(bound_g=nonpositive_sin_bounds())
    def test_nonincreasing_nonpositive_sin(self, mock_cos_visitor, bound_g):
        mono = mock_cos_visitor('nonincreasing', bound_g)
        assert mono.is_nonincreasing()


@pytest.fixture
def mock_pow_constant_base_visitor():
    def _f(base, mono_e, bound_e):
        v = MockMonotonicityPropagationVisitor()
        base = dex.Constant(base)
        v(base)
        v.add_bound(base, Bound(base.value, base.value))
        expo = v.add_mono(mono_e)
        v.add_bound(expo, bound_e)
        p = dex.PowExpression([base, expo])
        v(p)
        return v.get(p)
    return _f


class TestPowConstantBase(object):
    @pytest.mark.parametrize(
        'mono_e,bound_e',
        itertools.product(
            ['nondecreasing', 'nonincreasing', 'unknown'],
            ['nonpositive', 'nonnegative', 'unbounded']
        )
    )
    @given(base=reals(max_value=-0.01, allow_infinity=False))
    def test_negative_base(self, mock_pow_constant_base_visitor, base, mono_e, bound_e):
        mono = mock_pow_constant_base_visitor(base, mono_e, bound_e)
        assert mono == Monotonicity.Unknown

    @pytest.mark.parametrize('mono_e,bound_e,expected', [
        ('nondecreasing', 'nonpositive', Monotonicity.Nondecreasing),
        ('nondecreasing', 'nonnegative', Monotonicity.Unknown),
        ('nonincreasing', 'nonnegative', Monotonicity.Nondecreasing),
        ('nonincreasing', 'nonpositive', Monotonicity.Unknown),
    ])
    @given(base=reals(min_value=0.01, max_value=0.999))
    def test_base_between_0_and_1(self, mock_pow_constant_base_visitor, base,
                                  mono_e, bound_e, expected):
        mono = mock_pow_constant_base_visitor(base, mono_e, bound_e)
        assert mono == expected

    @pytest.mark.parametrize('mono_e,bound_e,expected', [
        ('nondecreasing', 'nonnegative', Monotonicity.Nondecreasing),
        ('nondecreasing', 'nonpositive', Monotonicity.Unknown),
        ('nonincreasing', 'nonpositive', Monotonicity.Nondecreasing),
        ('nonincreasing', 'nonnegative', Monotonicity.Unknown),
    ])
    @given(base=reals(min_value=1.01, allow_infinity=False))
    def test_base_gt_1(self, mock_pow_constant_base_visitor, base,
                       mono_e, bound_e, expected):
        mono = mock_pow_constant_base_visitor(base, mono_e, bound_e)
        assert mono == expected


@pytest.fixture
def mock_pow_constant_exponent_visitor():
    def _f(mono_b, bound_b, expo):
        v = MockMonotonicityPropagationVisitor()
        expo = dex.Constant(expo)
        v(expo)
        v.add_bound(expo, Bound(expo.value, expo.value))
        base = v.add_mono(mono_b)
        v.add_bound(base, bound_b)
        p = dex.PowExpression([base, expo])
        v(p)
        return v.get(p)
    return _f


class TestPowConstantExponent(object):
    @pytest.mark.parametrize(
        'mono_b,bound_b',
        itertools.product(['nonincreasing', 'nondecreasing'],
                          ['nonpositive', 'nonnegative', 'unbounded'])
    )
    def test_exponent_equals_1(self, mock_pow_constant_exponent_visitor, mono_b, bound_b):
        mono = mock_pow_constant_exponent_visitor(mono_b, bound_b, 1.0)
        assert mono == mono_description_to_mono(mono_b)

    @pytest.mark.parametrize(
        'mono_b,bound_b',
        itertools.product(['nonincreasing', 'nondecreasing'],
                          ['nonpositive', 'nonnegative', 'unbounded'])
    )
    def test_exponent_equals_0(self, mock_pow_constant_exponent_visitor, mono_b, bound_b):
        mono = mock_pow_constant_exponent_visitor(mono_b, bound_b, 0.0)
        assert mono == Monotonicity.Constant

    @pytest.mark.parametrize('mono_b,bound_b,expected', [
        ('nondecreasing', 'nonnegative', Monotonicity.Nondecreasing),
        ('nonincreasing', 'nonpositive', Monotonicity.Nondecreasing),

        ('nondecreasing', 'nonpositive', Monotonicity.Nonincreasing),
        ('nonincreasing', 'nonnegative', Monotonicity.Nonincreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_even_integer(self, mock_pow_constant_exponent_visitor, expo, mono_b, bound_b, expected):
        mono = mock_pow_constant_exponent_visitor(mono_b, bound_b, expo*2)
        assert mono == expected

    @pytest.mark.parametrize('mono_b,bound_b,expected', [
        ('nondecreasing', 'nonnegative', Monotonicity.Nonincreasing),
        ('nonincreasing', 'nonpositive', Monotonicity.Nonincreasing),

        ('nondecreasing', 'nonpositive', Monotonicity.Nondecreasing),
        ('nonincreasing', 'nonnegative', Monotonicity.Nondecreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_even_integer(self, mock_pow_constant_exponent_visitor, expo, mono_b, bound_b, expected):
        mono = mock_pow_constant_exponent_visitor(mono_b, bound_b, -expo*2)
        assert mono == expected

    @pytest.mark.parametrize('mono_b,expected', [
        ('nondecreasing', Monotonicity.Nondecreasing),
        ('nonincreasing', Monotonicity.Nonincreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_odd_integer(self, mock_pow_constant_exponent_visitor,
                                  expo, mono_b, expected):
        mono = mock_pow_constant_exponent_visitor(mono_b, 'unbounded', expo*2+1)
        assert mono == expected

    @pytest.mark.parametrize('mono_b,expected', [
        ('nondecreasing', Monotonicity.Nonincreasing),
        ('nonincreasing', Monotonicity.Nondecreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_odd_integer(self, mock_pow_constant_exponent_visitor,
                                  expo, mono_b, expected):
        mono = mock_pow_constant_exponent_visitor(mono_b, 'unbounded', -expo*2+1)
        assert mono == expected


@pytest.fixture
def mock_pow_visitor():
    def _f(mono_b, bound_b, mono_e, bound_e):
        v = MockMonotonicityPropagationVisitor()
        b = v.add_mono(mono_b)
        v.add_bound(b, bound_b)
        e = v.add_mono(mono_e)
        v.add_bound(e, bound_e)
        p = dex.PowExpression([b, e])
        v(p)
        return v.get(p)
    return _f


@pytest.mark.parametrize('mono_b,bound_b,mono_e,bound_e',
    itertools.product(
        ['nonincreasing', 'nondecreasing', 'unknown'],
        ['nonpositive', 'nonnegative', 'unbounded'],
        ['nonincreasing', 'nondecreasing', 'unknown'],
        ['nonpositive', 'nonnegative', 'unbounded']
    )
)
def test_pow(mock_pow_visitor, mono_b, bound_b, mono_e, bound_e):
    mono = mock_pow_visitor(mono_b, bound_b, mono_e, bound_e)
    assert mono == Monotonicity.Unknown

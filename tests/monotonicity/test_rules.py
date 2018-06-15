# pylint: skip-file
import pytest
from hypothesis import assume, given
import hypothesis.strategies as st
import itertools
from tests.conftest import coefficients, reals
from tests.conftest import PlaceholderExpression as PE
from suspect.expression import ExpressionType as ET, UnaryFunctionType as UFT
from suspect.monotonicity.monotonicity import Monotonicity as M
from suspect.interval import Interval as I
from suspect.monotonicity.rules import *
from suspect.math import almosteq, pi


class MonotonicityContext:
    def __init__(self, ctx=None, bounds=None):
        if ctx is None:
            ctx = {}
        if bounds is None:
            bounds = {}
        self.ctx = ctx
        self._bounds = bounds

    def monotonicity(self, expr):
        return self.ctx[expr]

    def bounds(self, expr):
        return self._bounds[expr]


@st.composite
def nondecreasing_terms(draw):
    coef = draw(coefficients())
    if coef > 0:
        return coef, M.Nondecreasing
    return coef, M.Nonincreasing


@st.composite
def nonincreasing_terms(draw):
    coef = draw(coefficients())
    if coef < 0:
        return coef, M.Nondecreasing
    return coef, M.Nonincreasing


@st.composite
def constant_terms(draw):
    coef = draw(coefficients())
    return coef, M.Constant


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
    b = I(start + mul, start + end + mul)
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
    b = I(start + mul, start + end + mul)
    assume(b.size() > 0)
    return b


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
    b = I(start + mul, start + end + mul)
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
    b = I(start + mul, start + end + mul)
    assume(b.size() > 0)
    return b


def test_variable_is_nondecreasing():
    rule = VariableRule()
    ctx = MonotonicityContext()
    result = rule.checked_apply(PE(ET.Variable), ctx)
    assert result.is_nondecreasing() and (not result.is_constant())


def test_constant_is_constant():
    rule = ConstantRule()
    ctx = MonotonicityContext()
    result = rule.checked_apply(PE(ET.Constant), ctx)
    assert result.is_constant()


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nondecreasing),
    (M.Nonincreasing, M.Nonincreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
def test_lte_constraint(expr_mono, expected_mono):
    rule = ConstraintRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: expr_mono})
    result = rule.checked_apply(PE(ET.Constraint, [child], bounded_above=True), ctx)
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nonincreasing),
    (M.Nonincreasing, M.Nondecreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
def test_gte_constraint(expr_mono, expected_mono):
    rule = ConstraintRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: expr_mono})
    result = rule.checked_apply(PE(ET.Constraint, [child], bounded_below=True), ctx)
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Unknown),
    (M.Nonincreasing, M.Unknown),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
def test_eq_constraint(expr_mono, expected_mono):
    rule = ConstraintRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: expr_mono})
    result = rule.checked_apply(PE(ET.Constraint, [child], bounded_above=True, bounded_below=True), ctx)
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nondecreasing),
    (M.Nonincreasing, M.Nonincreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
def test_min_constraint(expr_mono, expected_mono):
    rule = ObjectiveRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: expr_mono})
    result = rule.checked_apply(PE(ET.Objective, [child], is_minimizing=True), ctx)
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nonincreasing),
    (M.Nonincreasing, M.Nondecreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
def test_max_constraint(expr_mono, expected_mono):
    rule = ObjectiveRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: expr_mono})
    result = rule.checked_apply(PE(ET.Objective, [child], is_minimizing=False), ctx)
    assert result == expected_mono


@pytest.mark.parametrize('mono_f,mono_g,bound_f,bound_g,expected', [
    (M.Nondecreasing, M.Constant, I(None, None), I(0, None), M.Nondecreasing),
    (M.Nonincreasing, M.Constant, I(None, None), I(None, 0), M.Nondecreasing),

    (M.Nondecreasing, M.Constant, I(None, None), I(None, 0), M.Nonincreasing),
    (M.Nonincreasing, M.Constant, I(None, None), I(0, None), M.Nonincreasing),

    (M.Constant, M.Nondecreasing, I(0, None), I(None, None), M.Nondecreasing),
    (M.Constant, M.Nonincreasing, I(None, 0), I(None, None), M.Nondecreasing),

    (M.Constant, M.Nondecreasing, I(None, 0), I(None, None), M.Nonincreasing),
    (M.Constant, M.Nonincreasing, I(0, None), I(None, None), M.Nonincreasing),

    (M.Constant, M.Constant, I(None, None), I(None, None), M.Constant),

    (M.Nondecreasing, M.Nondecreasing, I(0, None), I(0, None), M.Nondecreasing),
    (M.Nondecreasing, M.Nonincreasing, I(None, 0), I(0, None), M.Nondecreasing),
    (M.Nonincreasing, M.Nondecreasing, I(0, None), I(None, 0), M.Nondecreasing),
    (M.Nonincreasing, M.Nonincreasing, I(None, 0), I(None, 0), M.Nondecreasing),

    (M.Nonincreasing, M.Nonincreasing, I(0, None), I(0, None), M.Nonincreasing),
    (M.Nonincreasing, M.Nondecreasing, I(None, 0), I(0, None), M.Nonincreasing),
    (M.Nondecreasing, M.Nonincreasing, I(0, None), I(None, 0), M.Nonincreasing),
    (M.Nondecreasing, M.Nondecreasing, I(None, 0), I(None, 0), M.Nonincreasing),
])
def test_product(mono_f, mono_g, bound_f, bound_g, expected):
    rule = ProductRule()
    f = PE(ET.UnaryFunction)
    g = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({f: mono_f, g: mono_g}, {f: bound_f, g: bound_g})
    result = rule.checked_apply(PE(ET.Product, [f, g]), ctx)
    assert result == expected


@pytest.mark.parametrize('mono_f,mono_g,bound_f,bound_g,expected', [
    (M.Constant, M.Constant, I(1.0, 1.0), I(2.0, 2.0), M.Constant),
    (M.Constant, M.Constant, I(1.0, 1.0), I(0.0, 0.0), M.Unknown),

    (M.Nondecreasing, M.Constant, I(None, None), I(10.0, 10.0), M.Nondecreasing),
    (M.Nonincreasing, M.Constant, I(None, None), I(-1.0, -1.0), M.Nondecreasing),

    (M.Nondecreasing, M.Constant, I(None, None), I(-1.0, -1.0), M.Nonincreasing),
    (M.Nonincreasing, M.Constant, I(None, None), I(10.0, 10.0), M.Nonincreasing),

    (M.Nondecreasing, M.Nonincreasing, I(0, None), I(0, None), M.Nondecreasing),
    (M.Nondecreasing, M.Nondecreasing, I(None, 0), I(0, None), M.Nondecreasing),
    (M.Nonincreasing, M.Nonincreasing, I(0, None), I(None, 0), M.Nondecreasing),
    (M.Nonincreasing, M.Nondecreasing, I(None, 0), I(None, 0), M.Nondecreasing),

    (M.Nonincreasing, M.Nondecreasing, I(0, None), I(0, None), M.Nonincreasing),
    (M.Nonincreasing, M.Nonincreasing, I(None, 0), I(0, None), M.Nonincreasing),
    (M.Nondecreasing, M.Nondecreasing, I(0, None), I(None, 0), M.Nonincreasing),
    (M.Nondecreasing, M.Nonincreasing, I(None, 0), I(None, 0), M.Nonincreasing),
])
def test_division(mono_f, mono_g, bound_f, bound_g, expected):
    rule = DivisionRule()
    f = PE(ET.UnaryFunction)
    g = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({f: mono_f, g: mono_g}, {f: bound_f, g: bound_g})
    result = rule.checked_apply(PE(ET.Division, [f, g]), ctx)
    assert result == expected



class TestLinear(object):
    def _result_with_terms(self, terms):
        rule = LinearRule()
        monotonicity = {}
        coefficients = [c for c, _ in terms]
        children = [PE(ET.Variable) for _ in terms]
        monos = [m for _, m in terms]
        for child, mono in zip(children, monos):
            monotonicity[child] = mono
        ctx = MonotonicityContext(monotonicity, {})
        result = rule.checked_apply(
            PE(ET.Linear, children=children, coefficients=coefficients),
            ctx,
        )
        return result

    @given(
        st.lists(
            st.one_of(nondecreasing_terms(), constant_terms()),
            min_size=1)
    )
    def test_nondecreasing(self, terms):
        mono = self._result_with_terms(terms)
        assert mono.is_nondecreasing()

    @given(
        st.lists(
            st.one_of(nonincreasing_terms(), constant_terms()),
            min_size=1)
    )
    def test_nonincreasing(self, terms):
        mono = self._result_with_terms(terms)
        assert mono.is_nonincreasing()

    @given(st.lists(constant_terms(), min_size=1))
    def test_constant(self, terms):
        mono = self._result_with_terms(terms)
        assert mono.is_constant()

    @given(
        st.lists(nondecreasing_terms(), min_size=1),
        st.lists(nonincreasing_terms(), min_size=1),
    )
    def test_unknown(self, a, b):
        terms = a + b
        for c, _ in terms:
            assume(c != 0.0)
        mono = self._result_with_terms(terms)
        assert mono.is_unknown()



class TestSum(object):
    def _result_with_terms(self, monos):
        rule = SumRule()
        monotonicity = {}
        children = [PE(ET.UnaryFunction) for _ in monos]
        for child, mono in zip(children, monos):
            monotonicity[child] = mono
        ctx = MonotonicityContext(monotonicity, {})
        result = rule.checked_apply(PE(ET.Sum, children=children), ctx)
        return result

    @given(
        st.lists(st.just(M.Nondecreasing), min_size=1),
        st.lists(st.one_of(st.just(M.Nondecreasing), st.just(M.Constant))),
    )
    def test_nondecreasing(self, a, b):
        mono = self._result_with_terms(a + b)
        assert mono.is_nondecreasing()

    @given(
        st.lists(st.just(M.Nonincreasing), min_size=1),
        st.lists(st.one_of(st.just(M.Nonincreasing), st.just(M.Constant))),
    )
    def test_nonincreasing(self, a, b):
        mono = self._result_with_terms(a + b)
        assert mono.is_nonincreasing()

    @given(st.lists(st.just(M.Constant), min_size=1))
    def test_constant(self, a):
        mono = self._result_with_terms(a)
        assert mono.is_constant()

    @given(
        st.lists(st.just(M.Nondecreasing), min_size=1),
        st.lists(st.just(M.Nonincreasing), min_size=1),
    )
    def test_unknown(self, a, b):
        mono = self._result_with_terms(a + b)
        assert mono.is_unknown()


@pytest.mark.parametrize('mono,bounds,expected', [
    (M.Nondecreasing, I(0, None), M.Nondecreasing),
    (M.Nonincreasing, I(None, 0), M.Nondecreasing),
    (M.Nonincreasing, I(0, None), M.Nonincreasing),
    (M.Nondecreasing, I(None, 0), M.Nonincreasing),
    (M.Constant, I(None, None), M.Constant),
    (M.Nondecreasing, I(None, None), M.Unknown),
    (M.Nonincreasing, I(None, None), M.Unknown),
])
def test_abs(mono, bounds, expected):
    rule = AbsRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: mono}, {child: bounds})
    result = rule.checked_apply(PE(ET.UnaryFunction, [child], func_type=UFT.Abs), ctx)
    assert result == expected


@pytest.mark.parametrize('rule_cls,func_type', [
    (SqrtRule, UFT.Sqrt), (ExpRule, UFT.Exp), (LogRule, UFT.Log),
    (TanRule, UFT.Tan), (AsinRule, UFT.Asin), (AtanRule, UFT.Atan)
])
@pytest.mark.parametrize('mono,bounds', itertools.product(
    [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
    [I(None, None), I(0, None), I(None, 0)],
))
def test_nondecreasing_function(rule_cls, func_type, mono, bounds):
    rule = rule_cls()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: mono}, {child: bounds})
    result = rule.checked_apply(PE(ET.UnaryFunction, [child], func_type=func_type), ctx)
    assert result == mono


@pytest.mark.parametrize('rule_cls,func_type', [(AcosRule, UFT.Acos)])
@pytest.mark.parametrize('mono,bounds', itertools.product(
    [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
    [I(None, None), I(0, None), I(None, 0)],
))
def test_nonincreasing_function(rule_cls, func_type, mono, bounds):
    rule = rule_cls()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: mono}, {child: bounds})
    result = rule.checked_apply(PE(ET.UnaryFunction, [child], func_type=func_type), ctx)
    assert result == mono.negate()


@pytest.mark.parametrize('mono,bounds', itertools.product(
    [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
    [I(None, None), I(0, None), I(None, 0)],
))
def test_negation(mono, bounds):
    rule = NegationRule()
    child = PE(ET.UnaryFunction)
    ctx = MonotonicityContext({child: mono}, {child: bounds})
    result = rule.checked_apply(PE(ET.Negation, [child]), ctx)
    assert result == mono.negate()


class TestPowConstantBase(object):
    def _result_with_base_expo(self, base, mono_expo, bounds_expo):
        base = PE(ET.Constant, is_constant=True, value=base)
        expo = PE()
        rule = PowerRule()
        ctx = MonotonicityContext({
            base: M.Constant,
            expo: mono_expo,
        }, {
            base: I(base.value, base.value),
            expo: bounds_expo,
        })
        return rule.checked_apply(PE(ET.Power, [base, expo]), ctx)

    @pytest.mark.parametrize(
        'mono_expo,bounds_expo',
        itertools.product(
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)]
        )
    )
    @given(base=reals(max_value=-0.01, allow_infinity=False))
    def test_negative_base(self, base, mono_expo, bounds_expo):
        mono = self._result_with_base_expo(base, mono_expo, bounds_expo)
        assert mono == M.Unknown

    @pytest.mark.parametrize('mono_expo,bounds_expo,expected', [
        (M.Nondecreasing, I(None, 0), M.Nondecreasing),
        (M.Nondecreasing, I(0, None), M.Unknown),
        (M.Nonincreasing, I(0, None), M.Nondecreasing),
        (M.Nonincreasing, I(None, 0), M.Unknown),
    ])
    @given(base=reals(min_value=0.01, max_value=0.999))
    def test_base_between_0_and_1(self, base, mono_expo, bounds_expo, expected):
        mono = self._result_with_base_expo(base, mono_expo, bounds_expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_expo,bounds_expo,expected', [
        (M.Nondecreasing, I(0, None), M.Nondecreasing),
        (M.Nondecreasing, I(None, 0), M.Unknown),
        (M.Nonincreasing, I(None, 0), M.Nondecreasing),
        (M.Nonincreasing, I(0, None), M.Unknown),
    ])
    @given(base=reals(min_value=1.01, allow_infinity=False))
    def test_base_gt_1(self, base, mono_expo, bounds_expo, expected):
        mono = self._result_with_base_expo(base, mono_expo, bounds_expo)
        assert mono == expected


class TestPowConstantExponent(object):
    def _result_with_base_expo(self, mono_base, bounds_base, expo):
        base = PE()
        expo = PE(ET.Constant, is_constant=True, value=expo)
        rule = PowerRule()
        ctx = MonotonicityContext({
            base: mono_base,
            expo: M.Constant,
        }, {
            base: bounds_base,
            expo: I(expo.value, expo.value),
        })
        return rule.checked_apply(PE(ET.Power, [base, expo]), ctx)

    @pytest.mark.parametrize(
        'mono_base,bounds_base',
        itertools.product([M.Nonincreasing, M.Nondecreasing],
                          [I(None, 0), I(0, None), I(None, None)])
    )
    def test_exponent_equals_1(self, mono_base, bounds_base):
        mono = self._result_with_base_expo(mono_base, bounds_base, 1.0)
        assert mono == mono_base

    @pytest.mark.parametrize(
        'mono_base,bounds_base',
        itertools.product([M.Nonincreasing, M.Nondecreasing],
                          [I(None, 0), I(0, None), I(None, None)])
    )
    def test_exponent_equals_0(self, mono_base, bounds_base):
        mono = self._result_with_base_expo(mono_base, bounds_base, 0.0)
        assert mono == M.Constant

    @pytest.mark.parametrize('mono_base,bounds_base,expected', [
        (M.Nondecreasing, I(0, None), M.Nondecreasing),
        (M.Nonincreasing, I(None, 0), M.Nondecreasing),

        (M.Nondecreasing, I(None, 0), M.Nonincreasing),
        (M.Nonincreasing, I(0, None), M.Nonincreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_even_integer(self, expo, mono_base, bounds_base, expected):
        mono = self._result_with_base_expo(mono_base, bounds_base, 2*expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,bounds_base,expected', [
        (M.Nondecreasing, I(0, None), M.Nonincreasing),
        (M.Nonincreasing, I(None, 0), M.Nonincreasing),

        (M.Nondecreasing, I(None, 0), M.Nondecreasing),
        (M.Nonincreasing, I(0, None), M.Nondecreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_even_integer(self, expo, mono_base, bounds_base, expected):
        mono = self._result_with_base_expo(mono_base, bounds_base, -2*expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nondecreasing, M.Nondecreasing),
        (M.Nonincreasing, M.Nonincreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_odd_integer(self, expo, mono_base, expected):
        mono = self._result_with_base_expo(mono_base, I(None, None), 2*expo+1)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nondecreasing, M.Nonincreasing),
        (M.Nonincreasing, M.Nondecreasing),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_odd_integer(self, expo, mono_base, expected):
        mono = self._result_with_base_expo(mono_base, I(None, None), -2*expo+1)
        assert mono == expected

    @pytest.mark.parametrize('mono_base', [M.Nondecreasing, M.Nondecreasing])
    @given(expo=reals(allow_infinity=False))
    def test_noninteger_negative_base(self, expo, mono_base):
        assume(not almosteq(expo,  0))
        assume(expo != int(expo))
        mono = self._result_with_base_expo(mono_base, I(None, 0), expo)
        assert mono == M.Unknown

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nondecreasing, M.Nondecreasing),
        (M.Nonincreasing, M.Nonincreasing)
    ])
    @given(expo=reals(allow_infinity=False, min_value=1e-5))
    def test_positive_noninteger(self, expo, mono_base, expected):
        mono = self._result_with_base_expo(mono_base, I(0, None), expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nonincreasing, M.Nondecreasing),
        (M.Nondecreasing, M.Nonincreasing)
    ])
    @given(expo=reals(allow_infinity=False, max_value=-1e-5))
    def test_negative_noninteger(self, expo, mono_base, expected):
        mono = self._result_with_base_expo(mono_base, I(0, None), expo)
        assert mono == expected

@pytest.mark.parametrize('mono_base,bounds_base,mono_expo,bounds_expo',
    itertools.product(
        [M.Nonincreasing, M.Nondecreasing, M.Unknown],
        [I(None, 0), I(0, None), I(None, None)],
        [M.Nonincreasing, M.Nondecreasing, M.Unknown],
        [I(None, 0), I(0, None), I(None, None)],
    )
)
def test_pow(mono_base, bounds_base, mono_expo, bounds_expo):
    base = PE()
    expo = PE()
    rule = PowerRule()
    ctx = MonotonicityContext({
        base: mono_base,
        expo: mono_expo,
    }, {
        base: bounds_base,
        expo: bounds_expo,
    })
    mono = rule.checked_apply(PE(ET.Power, [base, expo]), ctx)
    assert mono == M.Unknown


class TestSin(object):
    def _result_with_mono_bounds(self, mono, bounds):
        rule = SinRule()
        child = PE()
        ctx = MonotonicityContext({child: mono}, {child: bounds})
        return rule.checked_apply(PE(ET.UnaryFunction, [child], func_type=UFT.Sin), ctx)

    @given(nonnegative_cos_bounds())
    def test_nondecreasing_nonnegative_cos(self, bounds):
        mono = self._result_with_mono_bounds(M.Nondecreasing, bounds)
        assert mono.is_nondecreasing()

    @given(nonnegative_cos_bounds())
    def test_nonincreasing_nonnegative_cos(self, bounds):
        mono = self._result_with_mono_bounds(M.Nonincreasing, bounds)
        assert mono.is_nonincreasing()

    @given(nonpositive_cos_bounds())
    def test_decreasing_nonpositive_cos(self, bounds):
        mono = self._result_with_mono_bounds(M.Nondecreasing, bounds)
        assert mono.is_nonincreasing()

    @given(nonpositive_cos_bounds())
    def test_nonincreasing_nonpositive_cos(self, bounds):
        mono = self._result_with_mono_bounds(M.Nonincreasing, bounds)
        assert mono.is_nondecreasing()


class TestCos(object):
    def _result_with_mono_bounds(self, mono, bounds):
        rule = CosRule()
        child = PE()
        ctx = MonotonicityContext({child: mono}, {child: bounds})
        return rule.checked_apply(PE(ET.UnaryFunction, [child], func_type=UFT.Cos), ctx)

    @given(nonnegative_sin_bounds())
    def test_nonincreasing_nonnegative_sin(self, bounds):
        mono = self._result_with_mono_bounds(M.Nonincreasing, bounds)
        assert mono.is_nondecreasing()

    @given(nonnegative_sin_bounds())
    def test_nondecreasing_nonnegative_sin(self, bounds):
        mono = self._result_with_mono_bounds(M.Nondecreasing, bounds)
        assert mono.is_nonincreasing()

    @given(nonpositive_sin_bounds())
    def test_nondecreasing_nonpositive_sin(self, bounds):
        mono = self._result_with_mono_bounds(M.Nondecreasing, bounds)
        assert mono.is_nondecreasing()

    @given(nonpositive_sin_bounds())
    def test_nonincreasing_nonpositive_sin(self, bounds):
        mono = self._result_with_mono_bounds(M.Nonincreasing, bounds)
        assert mono.is_nonincreasing()

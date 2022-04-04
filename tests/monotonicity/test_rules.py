# pylint: skip-file
import pytest
from hypothesis import given, assume, reproduce_failure, settings, HealthCheck
import hypothesis.strategies as st
import itertools
from pyomo.core.kernel.component_map import ComponentMap
import pyomo.environ as pe
from tests.strategies import coefficients, reals, expressions
from tests.conftest import PlaceholderExpression as PE
from suspect.expression import ExpressionType as ET, UnaryFunctionType as UFT
from suspect.monotonicity.visitor import MonotonicityPropagationVisitor
from suspect.monotonicity.monotonicity import Monotonicity as M
from suspect.pyomo.expressions import (
    NumericConstant,
    ProductExpression,
    PowExpression,
    SumExpression,
    NegationExpression,
    DivisionExpression,
    Constraint,
    Objective,
    Sense,
)
from suspect.interval import Interval as I
from suspect.monotonicity.rules import *
from suspect.math import almosteq, pi


@pytest.fixture(scope='module')
def visitor():
    return MonotonicityPropagationVisitor()


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
    result = rule.apply(PE(ET.Variable), None, None)
    assert result.is_nondecreasing() and (not result.is_constant())


def test_constant_is_constant():
    rule = ConstantRule()
    result = rule.apply(PE(ET.Constant), None, None)
    assert result.is_constant()


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nondecreasing),
    (M.Nonincreasing, M.Nonincreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
@given(child=expressions())
def test_lte_constraint(visitor, child, expr_mono, expected_mono):
    mono = ComponentMap()
    mono[child] = expr_mono
    expr = Constraint('cons', None, 0.0, children=[child])
    matched, result = visitor.visit_expression(expr, mono, None)
    assert matched
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nonincreasing),
    (M.Nonincreasing, M.Nondecreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
@given(child=expressions())
def test_gte_constraint(visitor, child, expr_mono, expected_mono):
    mono = ComponentMap()
    mono[child] = expr_mono
    expr = Constraint('cons', 0.0, None, children=[child])
    matched, result = visitor.visit_expression(expr, mono, None)
    assert matched
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Unknown),
    (M.Nonincreasing, M.Unknown),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
@given(child=expressions())
def test_eq_constraint(visitor, child, expr_mono, expected_mono):
    mono = ComponentMap()
    mono[child] = expr_mono
    expr = Constraint('cons', 0.0, 0.0, children=[child])
    matched, result = visitor.visit_expression(expr, mono, None)
    assert matched
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nondecreasing),
    (M.Nonincreasing, M.Nonincreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
@given(child=expressions())
def test_min_constraint(visitor, child, expr_mono, expected_mono):
    mono = ComponentMap()
    mono[child] = expr_mono
    expr = Objective('obj', children=[child])
    matched, result = visitor.visit_expression(expr, mono, None)
    assert matched
    assert result == expected_mono


@pytest.mark.parametrize('expr_mono,expected_mono', [
    (M.Nondecreasing, M.Nonincreasing),
    (M.Nonincreasing, M.Nondecreasing),
    (M.Constant, M.Constant),
    (M.Unknown, M.Unknown),
])
@given(child=expressions())
def test_max_constraint(visitor, child, expr_mono, expected_mono):
    mono = ComponentMap()
    mono[child] = expr_mono
    expr = Objective('obj', children=[child], sense=Sense.MAXIMIZE)
    matched, result = visitor.visit_expression(expr, mono, None)
    assert matched
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
@given(f=expressions(), g=expressions())
def test_product(visitor, f, g, mono_f, mono_g, bound_f, bound_g, expected):
    bounds = ComponentMap()
    bounds[f] = bound_f
    bounds[g] = bound_g
    mono = ComponentMap()
    mono[f] = mono_f
    mono[g] = mono_g

    expr = f * g
    assume(isinstance(expr, ProductExpression))
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == expected


@pytest.mark.skip('Pyomo has no division expression')
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
@given(f=expressions(), g=expressions())
def test_division(visitor, f, g, mono_f, mono_g, bound_f, bound_g, expected):
    bounds = ComponentMap()
    bounds[f] = bound_f
    bounds[g] = bound_g
    mono = ComponentMap()
    mono[f] = mono_f
    mono[g] = mono_g

    expr = f / g
    # assume(isinstance(expr, DivisionExpression))
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == expected


@pytest.mark.parametrize('mono_g,bound_g,expected', [
    (M.Constant, I(2.0, 2.0), M.Constant),
    (M.Constant, I(0.0, 0.0), M.Unknown),
])
@given(g=expressions())
def test_division(visitor, g, mono_g, bound_g, expected):
    num = NumericConstant(1.0)
    bounds = ComponentMap()
    bounds[num] = I(1.0, 1.0)
    bounds[g] = bound_g
    mono = ComponentMap()
    mono[num] = M.Constant
    mono[g] = mono_g

    print(mono)
    print(mono[num])
    expr = DivisionExpression([num, g])
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == expected


@pytest.mark.skip('Pyomo has no linear expressions')
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
    def _result_with_terms(self, visitor, expr_with_monos):
        children = [c for c, _ in expr_with_monos]
        monos = [m for _, m in expr_with_monos]
        monotonicity = ComponentMap()
        for child, mono in expr_with_monos:
            monotonicity[child] = mono
        expr = SumExpression(children)
        matched, result = visitor.visit_expression(expr, monotonicity, None)
        assert matched
        return result

    @given(
        st.lists(
            st.tuples(expressions(), st.just(M.Nondecreasing)),
            min_size=1,
        ),
        st.lists(
            st.tuples(
                expressions(),
                st.one_of(st.just(M.Nondecreasing), st.just(M.Constant))
            )
        ),
    )
    def test_nondecreasing(self, visitor, a, b):
        mono = self._result_with_terms(visitor, a + b)
        assert mono.is_nondecreasing()

    @given(
        st.lists(
            st.tuples(expressions(), st.just(M.Nonincreasing)),
            min_size=1
        ),
        st.lists(
            st.tuples(
                expressions(),
                st.one_of(st.just(M.Nonincreasing), st.just(M.Constant))
            )
        ),
    )
    def test_nonincreasing(self, visitor, a, b):
        mono = self._result_with_terms(visitor, a + b)
        assert mono.is_nonincreasing()

    @given(
        st.lists(
            st.tuples(
                expressions(),
                st.just(M.Constant)
            ),
            min_size=1
        ),
    )
    def test_constant(self, visitor, a):
        mono = self._result_with_terms(visitor, a)
        assert mono.is_constant()

    @given(
        st.lists(
            st.tuples(
                expressions(),
                st.just(M.Nondecreasing)
            ),
            min_size=1
        ),
        st.lists(
            st.tuples(
                expressions(),
                st.just(M.Nonincreasing),
            ),
            min_size=1
        ),
    )
    def test_unknown(self, visitor, a, b):
        mono = self._result_with_terms(visitor, a + b)
        assert mono.is_unknown()


@pytest.mark.parametrize('mono_g,bounds_g,expected', [
    (M.Nondecreasing, I(0, None), M.Nondecreasing),
    (M.Nonincreasing, I(None, 0), M.Nondecreasing),
    (M.Nonincreasing, I(0, None), M.Nonincreasing),
    (M.Nondecreasing, I(None, 0), M.Nonincreasing),
    (M.Constant, I(None, None), M.Constant),
    (M.Nondecreasing, I(None, None), M.Unknown),
    (M.Nonincreasing, I(None, None), M.Unknown),
])
@given(g=expressions())
def test_abs(visitor, g, mono_g, bounds_g, expected):
    bounds = ComponentMap()
    bounds[g] = bounds_g
    mono = ComponentMap()
    mono[g] = mono_g
    expr = abs(g)
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == expected


@pytest.mark.parametrize('func_name', [
    'sqrt', 'exp', 'log', 'tan', 'asin', 'atan',
])
@pytest.mark.parametrize('mono_g,bounds_g', itertools.product(
    [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
    [I(None, None), I(0, None), I(None, 0)],
))
@given(g=expressions())
def test_nondecreasing_function(visitor, g, func_name, mono_g, bounds_g):
    mono = ComponentMap()
    mono[g] = mono_g
    bounds = ComponentMap()
    bounds[g]= bounds_g
    func = getattr(pe, func_name)
    expr = func(g)
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == mono_g


@pytest.mark.parametrize('func_name', ['acos'])
@pytest.mark.parametrize('mono_g,bounds_g', itertools.product(
    [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
    [I(None, None), I(0, None), I(None, 0)],
))
@given(g=expressions())
def test_nonincreasing_function(visitor, g, func_name, mono_g, bounds_g):
    mono = ComponentMap()
    mono[g] = mono_g
    bounds = ComponentMap()
    bounds[g] = bounds_g
    func = getattr(pe, func_name)
    expr = func(g)
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == mono_g.negate()


@pytest.mark.parametrize('mono_g,bounds_g', itertools.product(
    [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
    [I(None, None), I(0, None), I(None, 0)],
))
@given(g=expressions())
def test_negation(visitor, g, mono_g, bounds_g):
    mono = ComponentMap()
    mono[g] = mono_g
    bounds = ComponentMap()
    bounds[g] = bounds_g
    expr = -g
    assume(isinstance(expr, NegationExpression))
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == mono_g.negate()


class TestPowConstantBase(object):
    def _result_with_base_expo(self, visitor, base, expo, mono_expo, bounds_expo):
        rule = PowerRule()
        mono = ComponentMap()
        mono[base] = M.Constant
        mono[expo] = mono_expo
        bounds = ComponentMap()
        bounds[base] = I(base, base)
        bounds[expo] = bounds_expo
        expr = base ** expo
        assume(isinstance(expr, PowExpression))
        matched, result = visitor.visit_expression(expr, mono, bounds)
        assert matched
        return result

    @pytest.mark.parametrize(
        'mono_expo,bounds_expo',
        itertools.product(
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)]
        )
    )
    @given(
        base=reals(max_value=-0.01, allow_infinity=False),
        expo=expressions(),
    )
    def test_negative_base(self, visitor, base, expo, mono_expo, bounds_expo):
        mono = self._result_with_base_expo(visitor, base, expo, mono_expo, bounds_expo)
        assert mono == M.Unknown

    @pytest.mark.parametrize('mono_expo,bounds_expo,expected', [
        (M.Nondecreasing, I(None, 0), M.Nondecreasing),
        (M.Nondecreasing, I(0, None), M.Unknown),
        (M.Nonincreasing, I(0, None), M.Nondecreasing),
        (M.Nonincreasing, I(None, 0), M.Unknown),
    ])
    @given(
        base=reals(min_value=0.01, max_value=0.999),
        expo=expressions(),
    )
    def test_base_between_0_and_1(self, visitor, base, expo, mono_expo, bounds_expo, expected):
        mono = self._result_with_base_expo(visitor, base, expo, mono_expo, bounds_expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_expo,bounds_expo,expected', [
        (M.Nondecreasing, I(0, None), M.Nondecreasing),
        (M.Nondecreasing, I(None, 0), M.Unknown),
        (M.Nonincreasing, I(None, 0), M.Nondecreasing),
        (M.Nonincreasing, I(0, None), M.Unknown),
    ])
    @given(
        base=reals(min_value=1.01, allow_infinity=False),
        expo=expressions(),
    )
    def test_base_gt_1(self, visitor, base, expo, mono_expo, bounds_expo, expected):
        mono = self._result_with_base_expo(visitor, base, expo, mono_expo, bounds_expo)
        assert mono == expected


class TestPowConstantExponent(object):
    def _result_with_base_expo(self, visitor, base, mono_base, bounds_base, expo):
        mono = ComponentMap()
        mono[base] = mono_base
        mono[expo] = M.Constant
        bounds = ComponentMap()
        bounds[base] = bounds_base
        bounds[expo] = I(expo, expo)
        expr = PowExpression([base, expo])
        matched, result = visitor.visit_expression(expr, mono, bounds)
        assert matched
        return result

    @pytest.mark.parametrize(
        'mono_base,bounds_base',
        itertools.product([M.Nonincreasing, M.Nondecreasing],
                          [I(None, 0), I(0, None), I(None, None)])
    )
    @given(base=expressions())
    def test_exponent_equals_1(self, visitor, base, mono_base, bounds_base):
        mono = self._result_with_base_expo(visitor, base, mono_base, bounds_base, 1.0)
        assert mono == mono_base

    @pytest.mark.parametrize(
        'mono_base,bounds_base',
        itertools.product([M.Nonincreasing, M.Nondecreasing],
                          [I(None, 0), I(0, None), I(None, None)])
    )
    @given(base=expressions())
    def test_exponent_equals_0(self, visitor, base, mono_base, bounds_base):
        mono = self._result_with_base_expo(visitor, base, mono_base, bounds_base, 0.0)
        assert mono == M.Constant

    @pytest.mark.parametrize('mono_base,bounds_base,expected', [
        (M.Nondecreasing, I(0, None), M.Nondecreasing),
        (M.Nonincreasing, I(None, 0), M.Nondecreasing),

        (M.Nondecreasing, I(None, 0), M.Nonincreasing),
        (M.Nonincreasing, I(0, None), M.Nonincreasing),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000),
    )
    def test_positive_even_integer(self, visitor, base, expo, mono_base, bounds_base, expected):
        mono = self._result_with_base_expo(visitor, base, mono_base, bounds_base, 2*expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,bounds_base,expected', [
        (M.Nondecreasing, I(0, None), M.Nonincreasing),
        (M.Nonincreasing, I(None, 0), M.Nonincreasing),

        (M.Nondecreasing, I(None, 0), M.Nondecreasing),
        (M.Nonincreasing, I(0, None), M.Nondecreasing),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000)
    )
    def test_negative_even_integer(self, visitor, base, expo, mono_base, bounds_base, expected):
        mono = self._result_with_base_expo(visitor, base, mono_base, bounds_base, -2*expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nondecreasing, M.Nondecreasing),
        (M.Nonincreasing, M.Nonincreasing),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000)
    )
    def test_positive_odd_integer(self, visitor, base, expo, mono_base, expected):
        mono = self._result_with_base_expo(visitor, base, mono_base, I(None, None), 2*expo+1)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nondecreasing, M.Nonincreasing),
        (M.Nonincreasing, M.Nondecreasing),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000)
    )
    def test_negative_odd_integer(self, visitor, base, expo, mono_base, expected):
        mono = self._result_with_base_expo(
            visitor, base, mono_base, I(None, None), -2*expo+1
        )
        assert mono == expected

    @pytest.mark.parametrize('mono_base', [M.Nondecreasing, M.Nondecreasing])
    @given(
        base=expressions(),
        expo=reals(allow_infinity=False, min_value=-1e5, max_value=1e5),
    )
    def test_noninteger_negative_base(self, visitor, base, expo, mono_base):
        assume(not almosteq(expo,  0))
        assume(not almosteq(expo, int(expo)))
        mono = self._result_with_base_expo(
            visitor, base, mono_base, I(None, 0), expo
        )
        assert mono == M.Unknown

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nondecreasing, M.Nondecreasing),
        (M.Nonincreasing, M.Nonincreasing)
    ])
    @given(
        base=expressions(),
        expo=reals(allow_infinity=False, min_value=1e-5, max_value=1e-5)
    )
    def test_positive_noninteger(self, visitor, base, expo, mono_base, expected):
        mono = self._result_with_base_expo(visitor, base, mono_base, I(0, None), expo)
        assert mono == expected

    @pytest.mark.parametrize('mono_base,expected', [
        (M.Nonincreasing, M.Nondecreasing),
        (M.Nondecreasing, M.Nonincreasing)
    ])
    @given(
        base=expressions(),
        expo=reals(allow_infinity=False, min_value=-1e5, max_value=-1e-5)
    )
    def test_negative_noninteger(self, visitor, base, expo, mono_base, expected):
        mono = self._result_with_base_expo(visitor, base, mono_base, I(0, None), expo)
        assert mono == expected


@pytest.mark.parametrize('mono_base,bounds_base,mono_expo,bounds_expo',
    itertools.product(
        [M.Nonincreasing, M.Nondecreasing, M.Unknown],
        [I(None, 0), I(0, None), I(None, None)],
        [M.Nonincreasing, M.Nondecreasing, M.Unknown],
        [I(None, 0), I(0, None), I(None, None)],
    )
)
@given(base=expressions(), expo=expressions())
def test_pow(visitor, base, expo, mono_base, bounds_base, mono_expo, bounds_expo):
    mono = ComponentMap()
    mono[base] = mono_base
    mono[expo] = mono_expo
    bounds = ComponentMap()
    bounds[base] = bounds_base
    bounds[expo] = bounds_expo
    expr = PowExpression([base, expo])
    matched, result = visitor.visit_expression(expr, mono, bounds)
    assert matched
    assert result == M.Unknown


class TestSin(object):
    def _result_with_mono_bounds(self, visitor, g, mono_g, bounds_g):
        mono = ComponentMap()
        mono[g] = mono_g
        bounds = ComponentMap()
        bounds[g] = bounds_g
        expr = pe.sin(g)
        matched, result = visitor.visit_expression(expr, mono, bounds)
        assert matched
        return result

    @given(expressions(), nonnegative_cos_bounds())
    def test_nondecreasing_nonnegative_cos(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nondecreasing, bounds)
        assert mono.is_nondecreasing()

    @given(expressions(), nonnegative_cos_bounds())
    def test_nonincreasing_nonnegative_cos(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nonincreasing, bounds)
        assert mono.is_nonincreasing()

    @given(expressions(), nonpositive_cos_bounds())
    @pytest.mark.skip("Need investigation")
    def test_decreasing_nonpositive_cos(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nondecreasing, bounds)
        assert mono.is_nonincreasing()

    @given(expressions(), nonpositive_cos_bounds())
    def test_nonincreasing_nonpositive_cos(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nonincreasing, bounds)
        assert mono.is_nondecreasing()


class TestCos(object):
    def _result_with_mono_bounds(self, visitor, g, mono_g, bounds_g):
        mono = ComponentMap()
        mono[g] = mono_g
        bounds = ComponentMap()
        bounds[g] = bounds_g
        expr = pe.cos(g)
        matched, result = visitor.visit_expression(expr, mono, bounds)
        assert matched
        return result

    @given(expressions(), nonnegative_sin_bounds())
    def test_nonincreasing_nonnegative_sin(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nonincreasing, bounds)
        assert mono.is_nondecreasing()

    @given(expressions(), nonnegative_sin_bounds())
    def test_nondecreasing_nonnegative_sin(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nondecreasing, bounds)
        assert mono.is_nonincreasing()

    @given(expressions(), nonpositive_sin_bounds())
    def test_nondecreasing_nonpositive_sin(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nondecreasing, bounds)
        assert mono.is_nondecreasing()

    @given(expressions(), nonpositive_sin_bounds())
    def test_nonincreasing_nonpositive_sin(self, visitor, g, bounds):
        mono = self._result_with_mono_bounds(visitor, g, M.Nonincreasing, bounds)
        assert mono.is_nonincreasing()

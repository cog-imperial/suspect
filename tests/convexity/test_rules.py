# pylint: skip-file
import pytest
from hypothesis import assume, given, reproduce_failure
import hypothesis.strategies as st
import itertools
from tests.conftest import coefficients, reals
from tests.conftest import PlaceholderExpression as PE
from suspect.expression import ExpressionType as ET, UnaryFunctionType as UFT
from suspect.monotonicity import Monotonicity as M
from suspect.convexity import Convexity as C
from suspect.interval import Interval as I
from suspect.convexity.rules import *
from suspect.math import almosteq, pi


class ConvexityContext:
    def __init__(self, cvx=None, mono=None, bounds=None):
        if cvx is None:
            cvx = {}
        if mono is None:
            mono = {}
        if bounds is None:
            bounds = {}
        self._cvx = cvx
        self._mono = mono
        self._bounds = bounds

    def convexity(self, expr):
        return self._cvx[expr]

    def set_convexity(self, expr, value):
        self._cvx[expr] = value

    def monotonicity(self, expr):
        return self._mono[expr]

    def bounds(self, expr):
        return self._bounds[expr]


@st.composite
def convex_terms(draw):
    coef = draw(coefficients())
    if coef > 0:
        return (coef, C.Convex)
    else:
        return (coef, C.Concave)


@st.composite
def concave_terms(draw):
    coef = draw(coefficients())
    if coef < 0:
        return (coef, C.Convex)
    else:
        return (coef, C.Concave)


@st.composite
def linear_terms(draw):
    coef = draw(coefficients())
    return (coef, C.Linear)


def test_variable_is_linear():
    rule = VariableRule()
    ctx = ConvexityContext()
    result = rule.checked_apply(PE(ET.Variable), ctx)
    assert result.is_linear()


def test_constant_is_linear():
    rule = ConstantRule()
    ctx = ConvexityContext()
    result = rule.checked_apply(PE(ET.Constant), ctx)
    assert result.is_linear()


@pytest.mark.parametrize('cvx_child,bounded_below,bounded_above,expected', [
    # l <= g(x) <= u is always unknown except if g(x) is linear
    (C.Linear, True, True, C.Linear),
    (C.Convex, True, True, C.Unknown),
    (C.Concave, True, True, C.Unknown),
    (C.Unknown, True, True, C.Unknown),

    # g(x) <= u is the same as g(x)
    (C.Linear, False, True, C.Linear),
    (C.Convex, False, True, C.Convex),
    (C.Concave, False, True, C.Concave),
    (C.Unknown, False, True, C.Unknown),

    # l <= g(x) is the negation of g(x)
    (C.Linear, True, False, C.Linear),
    (C.Convex, True, False, C.Concave),
    (C.Concave, True, False, C.Convex),
    (C.Unknown, True, False, C.Unknown),
])
def test_constraint(cvx_child, bounded_below, bounded_above, expected):
    rule = ConstraintRule()
    child = PE()
    ctx = ConvexityContext({child: cvx_child})
    result = rule.checked_apply(
        PE(ET.Constraint, [child], bounded_below=bounded_below, bounded_above=bounded_above),
        ctx,
    )
    assert result == expected


@pytest.mark.parametrize('cvx_child,expected', [
    (C.Linear, C.Linear), (C.Convex, C.Convex), (C.Concave, C.Concave), (C.Unknown, C.Unknown),
])
def test_objective_is_minimizing(cvx_child, expected):
    rule = ObjectiveRule()
    child = PE()
    ctx = ConvexityContext({child: cvx_child})
    result = rule.checked_apply(
        PE(ET.Objective, [child], is_minimizing=True),
        ctx,
    )
    assert result == expected


@pytest.mark.parametrize('cvx_child,expected', [
    (C.Linear, C.Linear), (C.Convex, C.Concave), (C.Concave, C.Convex), (C.Unknown, C.Unknown),
])
def test_objective_is_maximizing(cvx_child, expected):
    rule = ObjectiveRule()
    child = PE()
    ctx = ConvexityContext({child: cvx_child})
    result = rule.checked_apply(
        PE(ET.Objective, [child], is_minimizing=False),
        ctx,
    )
    assert result == expected


@pytest.mark.parametrize('cvx,expected', [
    (C.Convex, C.Concave), (C.Concave, C.Convex), (C.Linear, C.Linear), (C.Unknown, C.Unknown),
])
def test_negation(cvx, expected):
    rule = NegationRule()
    child = PE()
    ctx = ConvexityContext(cvx={child: cvx})
    result = rule.checked_apply(PE(ET.Negation, [child]), ctx)
    assert result == expected


class TestProduct:
    def _rule_result(self, cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g):
        rule = ProductRule()
        f = PE()
        g = PE()
        ctx = ConvexityContext(
            cvx={f: cvx_f, g: cvx_g},
            mono={f: mono_f, g: mono_g},
            bounds={f: bounds_f, g: bounds_g},
        )
        return rule.checked_apply(PE(ET.Product, [f, g]), ctx)

    @pytest.mark.parametrize('cvx_f,bounds_g,expected', [
        (C.Convex, I(0, None), C.Convex),
        (C.Convex, I(None, 0), C.Concave),
        (C.Concave, I(0, None), C.Concave),
        (C.Concave, I(None, 0), C.Convex),
        (C.Linear, I(0, None), C.Linear),
        (C.Linear, I(None, 0), C.Linear),
        (C.Unknown, I(0, None), C.Unknown),
        (C.Unknown, I(None, 0), C.Unknown),
    ])
    def test_product_with_constant(self, cvx_f, bounds_g, expected):
        cvx_g = C.Linear # g is constant
        mono_f = M.Unknown
        mono_g = M.Constant
        bounds_f = I(None, None)
        assert self._rule_result(cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g) == expected
        assert self._rule_result(cvx_g, cvx_f, mono_g, mono_f, bounds_g, bounds_f) == expected

    @pytest.mark.parametrize('cvx_f,bounds_f,expected', [
        (C.Linear, I(None, None), C.Convex),
        (C.Linear, I(0, None), C.Convex),
        (C.Linear, I(None, 0), C.Convex),

        (C.Convex, I(None, None), C.Unknown),
        (C.Convex, I(0, None), C.Convex),
        (C.Convex, I(None, 0), C.Unknown),

        (C.Concave, I(None, None), C.Unknown),
        (C.Concave, I(0, None), C.Unknown),
        (C.Concave, I(None, 0), C.Convex),
    ])
    def test_product_with_itself(self, cvx_f, bounds_f, expected):
        rule = ProductRule()
        f = PE()
        ctx = ConvexityContext(
            cvx={f: cvx_f},
            bounds={f: bounds_f},
        )
        assert rule.checked_apply(PE(ET.Product, [f, f]), ctx) == expected

    @given(reals())
    def test_product_with_itself_with_coeff(self, coef):
        if coef > 0:
            expected = C.Convex
        else:
            expected = C.Concave
        rule = ProductRule()
        v = PE(ET.Variable)
        l = PE(ET.Linear, [v], coefficients=[coef])
        ctx = ConvexityContext()
        assert rule.checked_apply(PE(ET.Product, [l, v]), ctx) == expected
        assert rule.checked_apply(PE(ET.Product, [v, l]), ctx) == expected

    def test_product_linear_by_var(self):
        rule = ProductRule()
        v = PE(ET.Variable)
        l = PE(ET.Linear, [PE(ET.Variable), PE(ET.Variable)], coefficients=[1.0, 2.0])
        ctx = ConvexityContext(mono={v: M.Nondecreasing, l: M.Nondecreasing})
        assert rule.checked_apply(PE(ET.Product, [l, v]), ctx) == C.Unknown
        assert rule.checked_apply(PE(ET.Product, [v, l]), ctx) == C.Unknown


class TestDivision:
    def _rule_result(self, cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g):
        rule = DivisionRule()
        f = PE()
        g = PE()
        ctx = ConvexityContext(
            cvx={f: cvx_f, g: cvx_g},
            mono={f: mono_f, g: mono_g},
            bounds={f: bounds_f, g: bounds_g},
        )
        return rule.checked_apply(PE(ET.Division, [f, g]), ctx)

    @pytest.mark.parametrize('cvx_f,bounds_g,expected', [
        (C.Convex, I(1e-5, None), C.Convex),
        (C.Convex, I(0, None), C.Unknown),

        (C.Concave, I(None, -1e-5), C.Convex),
        (C.Concave, I(None, 0), C.Unknown),

        (C.Convex, I(None, -1e-5), C.Concave),
        (C.Convex, I(None, 0), C.Unknown),

        (C.Concave, I(1e-5, None), C.Concave),
        (C.Concave, I(0, None), C.Unknown),
    ])
    def test_constant_denominator(self, cvx_f, bounds_g, expected):
        assert self._rule_result(cvx_f, C.Linear, M.Unknown, M.Constant, I(None, None), bounds_g) == expected


    @pytest.mark.parametrize('cvx_g,bounds_f,bounds_g,expected', [
        (C.Concave, I(0, None), I(1e-5, None), C.Convex),
        (C.Concave, I(0, None), I(0, None), C.Unknown),
        (C.Convex, I(None, 0), I(None, -1e-5), C.Convex),
        (C.Convex, I(None, 0), I(None, 0), C.Unknown),
        (C.Convex, I(0, None), I(None, -1e-5), C.Concave),
        (C.Convex, I(0, None), I(None, 0), C.Unknown),
        (C.Concave, I(None, 0), I(1e-5, None), C.Concave),
        (C.Concave, I(None, 0), I(0, None), C.Unknown),
    ])
    def test_constant_numerator(self, cvx_g, bounds_f, bounds_g, expected):
        assert self._rule_result(C.Linear, cvx_g, M.Constant, M.Unknown, bounds_f, bounds_g) == expected


class TestLinear:
    def _rule_result(self, terms):
        coefficients = [c for c, _ in terms]
        children = [PE() for _ in terms]
        cvxs = [c for _, c in terms]
        ctx = ConvexityContext()
        for child, cvx in zip(children, cvxs):
            ctx.set_convexity(child, cvx)
        rule = LinearRule()
        return rule.checked_apply(PE(ET.Linear, children=children, coefficients=coefficients), ctx)

    @given(
        st.lists(convex_terms(), min_size=1),
        st.lists(linear_terms()),
    )
    def test_convex(self, a, b):
        cvx = self._rule_result(a + b)
        assert cvx.is_convex()

    @given(
        st.lists(concave_terms(), min_size=1),
        st.lists(linear_terms()),
    )
    def test_concave(self, a, b):
        cvx = self._rule_result(a + b)
        assert cvx.is_concave()

    @given(
        st.lists(linear_terms(), min_size=1),
    )
    def test_linear(self, a):
        cvx = self._rule_result(a)
        assert cvx.is_linear()

    @given(
        st.lists(convex_terms(), min_size=1),
        st.lists(concave_terms(), min_size=1),
    )
    def test_unknown(self, a, b):
        terms = a + b
        for c, _ in terms:
            assume(c != 0.0)
        cvx = self._rule_result(terms)
        assert cvx.is_unknown()


class TestSum:
    def _rule_result(self, cvxs):
        children = [PE() for _ in cvxs]
        ctx = ConvexityContext()
        for child, cvx in zip(children, cvxs):
            ctx.set_convexity(child, cvx)
        rule = SumRule()
        return rule.checked_apply(PE(ET.Sum, children=children), ctx)

    @given(
        st.lists(st.just(C.Convex), min_size=1),
        st.lists(st.just(C.Linear)),
    )
    def test_convex(self, a, b):
        cvx = self._rule_result(a + b)
        assert cvx.is_convex()

    @given(
        st.lists(st.just(C.Concave), min_size=1),
        st.lists(st.just(C.Linear)),
    )
    def test_concave(self, a, b):
        cvx = self._rule_result(a + b)
        assert cvx.is_concave()

    @given(
        st.lists(st.just(C.Linear), min_size=1),
    )
    def test_linear(self, a):
        cvx = self._rule_result(a)
        assert cvx.is_linear()

    @given(
        st.lists(st.just(C.Concave), min_size=1),
        st.lists(st.just(C.Convex), min_size=1),
    )
    def test_unknown(self, a, b):
        cvx = self._rule_result(a + b)
        assert cvx.is_unknown()


class UnaryFunctionTest:
    rule_cls = None
    func_type = None

    def _rule_result(self, cvx, mono, bounds):
        rule = self.rule_cls()
        child = PE()
        ctx = ConvexityContext({child: cvx}, {child: mono}, {child: bounds})
        return rule.checked_apply(PE(ET.UnaryFunction, [child], func_type=self.func_type), ctx)

class TestAbs(UnaryFunctionTest):
    rule_cls = AbsRule
    func_type = UFT.Abs

    @pytest.mark.parametrize('mono,bounds',
                             itertools.product([M.Nondecreasing, M.Nonincreasing, M.Constant],
                                               [I(None, 0), I(0, None), I(None, None)]))
    def test_linear_child(self, mono, bounds):
        cvx = self._rule_result(C.Linear, mono, bounds)
        assert cvx == C.Convex

    @pytest.mark.parametrize('mono,bounds,expected', [
        (M.Unknown, I(0, None), C.Convex),
        (M.Unknown, I(None, 0), C.Concave),
    ])
    def test_convex_child(self, mono, bounds, expected):
        cvx = self._rule_result(C.Convex, mono, bounds)
        assert cvx == expected

    @pytest.mark.parametrize('mono,bounds,expected', [
        (M.Unknown, I(0, None), C.Concave),
        (M.Unknown, I(None, 0), C.Convex),
    ])
    def test_concave_child(self, mono, bounds, expected):
        cvx = self._rule_result(C.Concave, mono, bounds)
        assert cvx == expected




class TestSqrt(UnaryFunctionTest):
    rule_cls = SqrtRule
    func_type = UFT.Sqrt

    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Concave, C.Linear],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        )
    )
    def test_concave_child(self, cvx, mono):
        cvx = self._rule_result(cvx, mono, I(0, None))
        assert cvx == C.Concave

    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Convex, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        ),
    )
    def test_non_concave_child(self, cvx, mono):
        cvx = self._rule_result(cvx, mono, I(0, None))
        assert cvx == C.Unknown


class TestExp(UnaryFunctionTest):
    rule_cls = ExpRule
    func_type = UFT.Exp

    @pytest.mark.parametrize(
        'cvx,mono,bounds',
        itertools.product(
            [C.Convex, C.Linear],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
            [I(0, None), I(None, 0), I(None, None)],
        )
    )
    def test_convex_child(self, cvx, mono, bounds):
        cvx = self._rule_result(cvx, mono, bounds)
        assert cvx == C.Convex

    @pytest.mark.parametrize(
        'cvx,mono,bounds',
        itertools.product(
            [C.Concave, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
            [I(0, None), I(None, 0), I(None, None)],
        )
    )
    def test_convex_child(self, cvx, mono, bounds):
        cvx = self._rule_result(cvx, mono, bounds)
        assert cvx == C.Unknown


class TestLog(UnaryFunctionTest):
    rule_cls = LogRule
    func_type = UFT.Log

    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Concave, C.Linear],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        )
    )
    def test_concave_child(self, cvx, mono):
        cvx = self._rule_result(cvx, mono, I(0, None))
        assert cvx == C.Concave

    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Convex, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        ),
    )
    def test_non_concave_child(self, cvx, mono):
        cvx = self._rule_result(cvx, mono, I(0, None))
        assert cvx == C.Unknown


class TestSin(UnaryFunctionTest):
    rule_cls = SinRule
    func_type = UFT.Sin

    def test_bound_size_too_big(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(-2, 2))
        assert cvx == C.Unknown

    def test_bound_opposite_sign(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(-0.1, 0.1))
        assert cvx == C.Unknown

    def test_positive_sin_linear_child(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(pi/2-0.1, pi/2+0.1))
        assert cvx == C.Concave

    def test_positive_sin_convex_child_but_wrong_interval(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(pi/2-0.1, pi/2+0.1))
        assert cvx == C.Unknown

    def test_positive_sin_convex_child(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(pi/2+0.1, pi-0.1))
        assert cvx == C.Concave

    def test_positive_sin_concave_child(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(0.1, pi/2-0.1))
        assert cvx == C.Concave

    def test_negative_sin_linear_child(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(1.5*pi-0.1, 1.5*pi+0.1))
        assert cvx == C.Convex

    def test_negative_sin_concave_child_but_wrong_interval(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(1.5*pi-0.1, 1.5*pi+0.1))
        assert cvx == C.Unknown

    def test_negative_sin_concave_child(self):
        cvx = self._rule_result(C.Concave, M.Constant, I(pi+0.1, 1.5*pi-0.1))
        assert cvx == C.Convex

    def test_negative_sin_convex_child(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(1.5*pi, 2*pi))
        assert cvx == C.Convex


class TestCos(UnaryFunctionTest):
    rule_cls = CosRule
    func_type = UFT.Cos

    def test_bound_size_too_big(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(-2, 2))
        assert cvx == C.Unknown

    def test_bound_opposite_sign(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(pi/2-0.1, pi/2+0.1))
        assert cvx == C.Unknown

    def test_positive_cos_linear_child(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(0, 0.5*pi))
        assert cvx == C.Concave

    def test_positive_cos_convex_child_but_wrong_interval(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(0, 0.5*pi))
        assert cvx == C.Unknown

    def test_positive_cos_convex_child(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(1.5*pi, 2*pi))
        assert cvx == C.Concave

    def test_positive_cos_concave_child(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(0, 0.5*pi))
        assert cvx == C.Concave

    def test_negative_cos_linear_child(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(0.6*pi, 1.4*pi))
        assert cvx == C.Convex

    def test_negative_cos_concave_child_but_wrong_interval(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(0.6*pi, 0.9*pi))
        assert cvx == C.Unknown

    def test_negative_cos_concave_child(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(1.1*pi, 1.4*pi))
        assert cvx == C.Convex

    def test_negative_cos_convex_child(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(0.6*pi, 0.9*pi))
        assert cvx == C.Convex



class TestTan(UnaryFunctionTest):
    rule_cls = TanRule
    func_type = UFT.Tan

    def test_bound_size_too_big(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(-2, 2))
        assert cvx == C.Unknown

    def test_bound_opposite_sign(self):
        cvx = self._rule_result(C.Linear, M.Constant, I(-0.1, 0.1))
        assert cvx == C.Unknown

    def test_positive_tan_convex_child(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(pi, 1.5*pi))
        assert cvx == C.Convex

    def test_positive_tan_concave_child(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(pi, 1.5*pi))
        assert cvx == C.Unknown

    def test_negative_tan_convex_child(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(-1.5*pi, -pi))
        assert cvx == C.Unknown

    def test_negative_tan_concave_child(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(-0.49*pi, -0.1))
        assert cvx == C.Concave


class TestAsin(UnaryFunctionTest):
    rule_cls = AsinRule
    func_type = UFT.Asin

    def test_is_concave(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(-1, 0))
        assert cvx == C.Concave

    @pytest.mark.parametrize('bounds', [I(-1, 0.1), I(-1.1, 0.0), I(None, None)])
    def test_is_not_concave(self, bounds):
        cvx = self._rule_result(C.Concave, M.Unknown, bounds)
        assert cvx == C.Unknown

    def test_is_convex(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(0, 1))
        assert cvx == C.Convex

    @pytest.mark.parametrize('bounds', [I(-0.1, 1), I(0.0, 1.1), I(None, None)])
    def test_is_not_convex(self, bounds):
        cvx = self._rule_result(C.Convex, M.Unknown, bounds)
        assert cvx == C.Unknown


class TestAcos(UnaryFunctionTest):
    rule_cls = AcosRule
    func_type = UFT.Acos

    def test_is_concave(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(0, 1))
        assert cvx == C.Concave

    @pytest.mark.parametrize('bounds', [I(-0.1, 1), I(0.0, 1.1), I(None, None)])
    def test_is_not_concave(self, bounds):
        cvx = self._rule_result(C.Convex, M.Unknown, bounds)
        assert cvx == C.Unknown

    def test_is_convex(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(-1, 0))
        assert cvx == C.Convex

    @pytest.mark.parametrize('bounds', [I(-1, 0.1), I(-1.1, 0.0), I(None, None)])
    def test_is_not_convex(self, bounds):
        cvx = self._rule_result(C.Concave, M.Unknown, bounds)
        assert cvx == C.Unknown


class TestAtan(UnaryFunctionTest):
    rule_cls = AtanRule
    func_type = UFT.Atan

    def test_is_concave(self):
        cvx = self._rule_result(C.Concave, M.Unknown, I(0, None))
        assert cvx == C.Concave

    @pytest.mark.parametrize('bounds', [I(None, 0), I(None, None)])
    def test_is_not_concave(self, bounds):
        cvx = self._rule_result(C.Concave, M.Unknown, bounds)
        assert cvx == C.Unknown

    def test_is_convex(self):
        cvx = self._rule_result(C.Convex, M.Unknown, I(None, 0))
        assert cvx == C.Convex

    @pytest.mark.parametrize('bounds', [I(0, None), I(None, None)])
    def test_is_not_convex(self, bounds):
        cvx = self._rule_result(C.Convex, C.Unknown, bounds)
        assert cvx == C.Unknown


class TestPowConstantBase:
    def _rule_result(self, base, cvx_expo, mono_expo, bounds_expo):
        base = PE(ET.Constant, value=base)
        expo = PE()
        ctx = ConvexityContext(
            cvx={expo: cvx_expo, base: C.Linear},
            mono={expo: mono_expo, base: M.Constant},
            bounds={base: I(base.value, base.value), expo: bounds_expo},
        )
        rule = PowerRule()
        return rule.checked_apply(PE(ET.Power, [base, expo]), ctx)

    @pytest.mark.parametrize(
        'cvx_expo,mono_expo,bounds_expo',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(base=reals(max_value=-0.01, allow_infinity=False))
    def test_negative_base(self, base, cvx_expo, mono_expo, bounds_expo):
        cvx = self._rule_result(base, cvx_expo, mono_expo, bounds_expo)
        assert cvx == C.Unknown

    @pytest.mark.parametrize('cvx_expo,mono_expo,bounds_expo',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(base=reals(min_value=0.001, max_value=0.999))
    def test_base_between_0_and_1(self, base, cvx_expo, mono_expo, bounds_expo):
        if cvx_expo == C.Concave or cvx_expo == C.Linear:
            expected = C.Convex
        else:
            expected = C.Unknown
        cvx = self._rule_result(base, cvx_expo, mono_expo, bounds_expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_expo,mono_expo,bounds_expo',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(base=reals(min_value=1, allow_infinity=False))
    def test_base_gt_1(self, base, cvx_expo, mono_expo, bounds_expo):
        if cvx_expo == C.Convex or cvx_expo == C.Linear:
            expected = C.Convex
        else:
            expected = C.Unknown
        cvx = self._rule_result(base, cvx_expo, mono_expo, bounds_expo)
        assert cvx == expected


class TestPowConstantExponent(object):
    def _rule_result(self, cvx_base, mono_base, bounds_base, expo):
        expo = PE(ET.Constant, value=expo)
        base = PE()
        ctx = ConvexityContext(
            cvx={base: cvx_base, expo: C.Linear},
            mono={base: mono_base, expo: M.Constant},
            bounds={base: bounds_base, expo: I(expo.value, expo.value)},
        )
        rule = PowerRule()
        return rule.checked_apply(PE(ET.Power, [base, expo]), ctx)

    @pytest.mark.parametrize(
        'cvx_base,mono_base,bounds_base',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    def test_exponent_equals_0(self, cvx_base, mono_base, bounds_base):
        cvx = self._rule_result(cvx_base, mono_base, bounds_base, 0.0)
        assert cvx == C.Linear

    @pytest.mark.parametrize(
        'cvx_base,mono_base,bounds_base',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    def test_exponent_equals_1(self, cvx_base, mono_base, bounds_base):
        cvx = self._rule_result(cvx_base, mono_base, bounds_base, 1.0)
        assert cvx == cvx_base

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Linear, M.Nondecreasing, I(None, None), C.Convex),

        (C.Convex, M.Unknown, I(0, None), C.Convex),
        (C.Convex, M.Unknown, I(None, 0), C.Unknown),

        (C.Concave, M.Unknown, I(0, None), C.Unknown),
        (C.Concave, M.Unknown, I(None, 0), C.Convex),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_even_integer(self, expo, cvx_base, mono_base, bounds_base, expected):
        cvx = self._rule_result(cvx_base, mono_base, bounds_base, 2*expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Convex, M.Unknown, I(None, 0), C.Convex),
        (C.Convex, M.Unknown, I(0, None), C.Concave),

        (C.Concave, M.Unknown, I(0, None), C.Convex),
        (C.Concave, M.Unknown, I(None, 0), C.Concave),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_even_integer(self, expo, cvx_base, mono_base, bounds_base, expected):
        cvx = self._rule_result(cvx_base, mono_base, bounds_base, -2*expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Convex, M.Unknown, I(0, None), C.Convex),
        (C.Convex, M.Unknown, I(None, 0), C.Unknown),
        (C.Concave, M.Unknown, I(None, 0), C.Concave),
        (C.Concave, M.Unknown, I(0, None), C.Unknown),
    ])
    @given(expo=st.integers(min_value=1))
    def test_positive_odd_integer(self, expo, cvx_base, mono_base, bounds_base, expected):
        cvx = self._rule_result(cvx_base, mono_base, bounds_base, 2*expo+1)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Concave, M.Unknown, I(0, None), C.Convex),
        (C.Concave, M.Unknown, I(None, 0), C.Unknown),
        (C.Convex, M.Unknown, I(None, 0), C.Concave),
        (C.Convex, M.Unknown, I(0, None), C.Unknown),
    ])
    @given(expo=st.integers(min_value=1))
    def test_negative_odd_integer(self, expo, cvx_base, mono_base, bounds_base, expected):
        cvx = self._rule_result(cvx_base, mono_base, bounds_base, -2*expo+1)
        assert cvx == expected

    @given(expo=reals(min_value=1, allow_infinity=False))
    def test_positive_gt_1_non_integer_negative_base(self, expo):
        expo = expo + 1e-6
        assume(expo != int(expo))
        cvx = self._rule_result(C.Convex, M.Unknown, I(None, -1), expo)
        assert cvx == C.Unknown

    @given(expo=reals(min_value=1, allow_infinity=False))
    def test_positive_gt_1_non_integer(self, expo):
        expo = expo + 1e-5 # make it positive
        assume(expo != int(expo))
        cvx = self._rule_result(C.Convex, M.Unknown, I(0, None), expo)
        assert cvx == C.Convex

    @pytest.mark.parametrize('cvx,expected', [(C.Convex, C.Concave), (C.Concave, C.Convex)])
    @given(expo=reals(max_value=0, allow_infinity=False))
    def test_positive_lt_0_non_integer(self, expo, cvx, expected):
        expo = expo - 1e-5 # make it negative
        assume(not almosteq(expo, int(expo)))
        cvx = self._rule_result(cvx, M.Unknown, I(0, None), expo)
        assert cvx == expected

    @given(expo=reals(min_value=0, max_value=1, allow_infinity=False))
    def test_positive_0_1_non_integer(self, expo):
        assume(not almosteq(expo, int(expo)))
        cvx = self._rule_result(C.Concave, M.Unknown, I(0, None), expo)
        assert cvx == C.Concave

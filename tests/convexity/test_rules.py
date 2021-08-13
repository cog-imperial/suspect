# pylint: skip-file
import pytest
from hypothesis import assume, given, reproduce_failure
import hypothesis.strategies as st
import itertools
import numpy as np
from pyomo.core.kernel.component_map import ComponentMap
import pyomo.environ as pe
from tests.strategies import coefficients, reals, variables, constants, expressions
from tests.conftest import PlaceholderExpression as PE, BilinearTerm
from suspect.expression import ExpressionType as ET, UnaryFunctionType as UFT
from suspect.monotonicity import Monotonicity as M
from suspect.convexity import Convexity as C
from suspect.convexity.visitor import ConvexityPropagationVisitor
from suspect.interval import Interval as I
from suspect.pyomo.expressions import *
from suspect.convexity.rules import *
from suspect.math import almosteq, pi


@pytest.fixture(scope='module')
def visitor():
    return ConvexityPropagationVisitor()


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


@given(variables())
def test_variable_is_linear(visitor, var):
    matched, result = visitor.visit_expression(var, None, None, None)
    assert matched
    assert result.is_linear()


@given(constants())
def test_constant_is_linear(visitor, const):
    matched, result = visitor.visit_expression(const, None, None, None)
    assert matched
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
@given(child=expressions())
def test_constraint(visitor, child, cvx_child, bounded_below, bounded_above, expected):
    if bounded_below:
        lower_bound = 0.0
    else:
        lower_bound = None

    if bounded_above:
        upper_bound = 1.0
    else:
        upper_bound = None

    expr = Constraint('constr', lower_bound, upper_bound, children=[child])

    convexity = ComponentMap()
    convexity[child] = cvx_child

    matched, result = visitor.visit_expression(expr, convexity, None, None)
    assert matched
    assert result == expected


@pytest.mark.parametrize('cvx_child,expected', [
    (C.Linear, C.Linear), (C.Convex, C.Convex), (C.Concave, C.Concave), (C.Unknown, C.Unknown),
])
@given(child=expressions())
def test_objective_is_minimizing(visitor, child, cvx_child, expected):
    expr = Objective('obj', children=[child])
    convexity = ComponentMap()
    convexity[child] = cvx_child

    matched, result = visitor.visit_expression(expr, convexity, None, None)
    assert matched
    assert result == expected


@pytest.mark.parametrize('cvx_child,expected', [
    (C.Linear, C.Linear), (C.Convex, C.Concave),
    (C.Concave, C.Convex), (C.Unknown, C.Unknown),
])
@given(child=expressions())
def test_objective_is_maximizing(visitor, child, cvx_child, expected):
    expr = Objective('obj', children=[child], sense=Sense.MAXIMIZE)
    convexity = ComponentMap()
    convexity[child] = cvx_child

    matched, result = visitor.visit_expression(expr, convexity, None, None)
    assert matched
    assert result == expected


@pytest.mark.parametrize('cvx,expected', [
    (C.Convex, C.Concave), (C.Concave, C.Convex),
    (C.Linear, C.Linear), (C.Unknown, C.Unknown),
])
@given(child=expressions())
def test_negation(visitor, child, cvx, expected):
    convexity = ComponentMap()
    convexity[child] = cvx
    expr = -child
    assume(isinstance(expr, NegationExpression))
    matched, result = visitor.visit_expression(expr, convexity, None, None)
    assert matched
    assert result == expected


class TestProduct:
    def _rule_result(self, visitor, f, g, cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g):
        convexity = ComponentMap()
        convexity[f] = cvx_f
        convexity[g] = cvx_g
        mono = ComponentMap()
        mono[f] = mono_f
        mono[g] = mono_g
        bounds = ComponentMap()
        bounds[f] = bounds_f
        bounds[g] = bounds_g
        expr = f * g
        assume(isinstance(expr, ProductExpression))
        matched, result = visitor.visit_expression(expr, convexity, mono, bounds)
        assert matched
        return result

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
    @given(f=expressions(), g=constants())
    def test_product_with_constant(self, visitor, f, g, cvx_f, bounds_g, expected):
        assume(f.is_expression_type() and not isinstance(f, MonomialTermExpression))
        cvx_g = C.Linear # g is constant
        mono_f = M.Unknown
        mono_g = M.Constant
        bounds_f = I(None, None)
        assert self._rule_result(visitor, f, g, cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g) == expected
        assert self._rule_result(visitor, g, f, cvx_g, cvx_f, mono_g, mono_f, bounds_g, bounds_f) == expected

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
    @given(f=expressions())
    def test_product_with_itself(self, visitor, f, cvx_f, bounds_f, expected):
        convexity = ComponentMap()
        convexity[f] = cvx_f
        bounds = ComponentMap()
        bounds[f] = bounds_f
        expr = f * f
        matched, result = visitor.visit_expression(expr, convexity, None, bounds)
        assert matched
        assert result == expected

    @given(var=variables(), coef=reals())
    def test_product_with_itself_with_coeff(self, visitor, var, coef):
        if coef > 0:
            expected = C.Convex
        else:
            expected = C.Concave
        rule = ProductRule()
        g = coef * var
        assume(isinstance(g, ProductExpression))
        matched, result = visitor.visit_expression(ProductExpression([var, g]), None, None, None)
        assert result == expected
        matched, result = visitor.visit_expression(ProductExpression([g, var]), None, None, None)
        assert result == expected

    @given(
        var=variables(),
        vars_with_coef=st.lists(
            st.tuples(
                variables(),
                constants(),
            ),
        )
    )
    def test_product_linear_by_var(self, visitor, var, vars_with_coef):
        rule = ProductRule()
        mono = ComponentMap()
        lin = sum(v * c for v, c in vars_with_coef)
        mono[var] = M.Nondecreasing
        mono[lin] = M.Nondecreasing
        matched, result = visitor.visit_expression(ProductExpression([var, lin]), None, mono, None)
        assert result == C.Unknown
        matched, result = visitor.visit_expression(ProductExpression([lin, var]), None, mono, None)
        assert result == C.Unknown


@pytest.mark.skip('Pyomo has no division')
class TestDivision:
    def _rule_result(self, cvx_f, cvx_g, mono_f, mono_g, bounds_f, bounds_g):
        rule = DivisionRule()
        f = PE()
        g = PE()
        convexity = ComponentMap()
        convexity[f] = cvx_f
        convexity[g] = cvx_g
        mono = ComponentMap()
        mono[f] = mono_f
        mono[g] = mono_g
        bounds = ComponentMap()
        bounds[f] = bounds_f
        bounds[g] = bounds_g
        return rule.apply(PE(ET.Division, [f, g]), convexity, mono, bounds)

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


@pytest.mark.skip('Pyomo has no linear expression')
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
    def _rule_result(self, visitor, children_with_cvx):
        children = [c for c, _ in children_with_cvx]

        convexity = ComponentMap()
        for child, cvx in children_with_cvx:
            convexity[child] = cvx
        expr = SumExpression(children)
        matched, result = visitor.visit_expression(expr, convexity, None, None)
        assert matched
        return result

    @given(
        st.lists(
            st.tuples(expressions(), st.just(C.Convex)),
            min_size=1
        ),
        st.lists(
            st.tuples(expressions(), st.just(C.Linear)),
        ),
    )
    def test_convex(self, visitor, a, b):
        cvx = self._rule_result(visitor, a + b)
        assert cvx.is_convex()

    @given(
        st.lists(
            st.tuples(expressions(), st.just(C.Concave)),
            min_size=1
        ),
        st.lists(
            st.tuples(expressions(), st.just(C.Linear))
        ),
    )
    def test_concave(self, visitor, a, b):
        cvx = self._rule_result(visitor, a + b)
        assert cvx.is_concave()

    @given(
        st.lists(
            st.tuples(expressions(), st.just(C.Linear)),
            min_size=1
        ),
    )
    def test_linear(self, visitor, a):
        cvx = self._rule_result(visitor, a)
        assert cvx.is_linear()

    @given(
        st.lists(
            st.tuples(expressions(), st.just(C.Concave)),
            min_size=1
        ),
        st.lists(
            st.tuples(expressions(), st.just(C.Convex)),
            min_size=1
        ),
    )
    def test_unknown(self, visitor, a, b):
        cvx = self._rule_result(visitor, a + b)
        assert cvx.is_unknown()


class UnaryFunctionTest:
    def _rule_result(self, visitor, g, cvx, mono_g, bounds_g, func):
        convexity = ComponentMap()
        convexity[g] = cvx
        mono = ComponentMap()
        mono[g] = mono_g
        bounds = ComponentMap()
        bounds[g] = bounds_g

        expr = func(g)
        matched, result = visitor.visit_expression(expr, convexity, mono, bounds)
        assert matched
        return result


class TestAbs(UnaryFunctionTest):
    @pytest.mark.parametrize('mono,bounds',
                             itertools.product([M.Nondecreasing, M.Nonincreasing, M.Constant],
                                               [I(None, 0), I(0, None), I(None, None)]))
    @given(child=expressions())
    def test_linear_child(self, visitor, child, mono, bounds):
        cvx = self._rule_result(visitor, child, C.Linear, mono, bounds, abs)
        assert cvx == C.Convex

    @pytest.mark.parametrize('mono,bounds,expected', [
        (M.Unknown, I(0, None), C.Convex),
        (M.Unknown, I(None, 0), C.Concave),
    ])
    @given(child=expressions())
    def test_convex_child(self, visitor, child, mono, bounds, expected):
        cvx = self._rule_result(visitor, child, C.Convex, mono, bounds, abs)
        assert cvx == expected

    @pytest.mark.parametrize('mono,bounds,expected', [
        (M.Unknown, I(0, None), C.Concave),
        (M.Unknown, I(None, 0), C.Convex),
    ])
    @given(child=expressions())
    def test_concave_child(self, visitor, child, mono, bounds, expected):
        cvx = self._rule_result(visitor, child, C.Concave, mono, bounds, abs)
        assert cvx == expected




class TestSqrt(UnaryFunctionTest):
    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Concave, C.Linear],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        )
    )
    @given(child=expressions())
    def test_concave_child(self, visitor, child, cvx, mono):
        cvx = self._rule_result(visitor, child, cvx, mono, I(0, None), pe.sqrt)
        assert cvx == C.Concave

    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Convex, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        ),
    )
    @given(child=expressions())
    def test_non_concave_child(self, visitor, child, cvx, mono):
        cvx = self._rule_result(visitor, child, cvx, mono, I(0, None), pe.sqrt)
        assert cvx == C.Unknown


class TestExp(UnaryFunctionTest):
    @pytest.mark.parametrize(
        'cvx,mono,bounds',
        itertools.product(
            [C.Convex, C.Linear],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
            [I(0, None), I(None, 0), I(None, None)],
        )
    )
    @given(child=expressions())
    def test_convex_child(self, visitor, child, cvx, mono, bounds):
        cvx = self._rule_result(visitor, child, cvx, mono, bounds, pe.exp)
        assert cvx == C.Convex

    @pytest.mark.parametrize(
        'cvx,mono,bounds',
        itertools.product(
            [C.Concave, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
            [I(0, None), I(None, 0), I(None, None)],
        )
    )
    @given(child=expressions())
    def test_convex_child(self, visitor, child, cvx, mono, bounds):
        cvx = self._rule_result(visitor, child, cvx, mono, bounds, pe.exp)
        assert cvx == C.Unknown


class TestLog(UnaryFunctionTest):
    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Concave, C.Linear],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        )
    )
    @given(child=expressions())
    def test_concave_child(self, visitor, child, cvx, mono):
        cvx = self._rule_result(visitor, child, cvx, mono, I(0, None), pe.log)
        assert cvx == C.Concave

    @pytest.mark.parametrize(
        'cvx,mono',
        itertools.product(
            [C.Convex, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Constant, M.Unknown],
        ),
    )
    @given(child=expressions())
    def test_non_concave_child(self, visitor, child, cvx, mono):
        cvx = self._rule_result(visitor, child, cvx, mono, I(0, None), pe.log)
        assert cvx == C.Unknown


class TestSin(UnaryFunctionTest):
    @given(expressions())
    def test_bound_size_too_big(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(-2, 2), pe.sin)
        assert cvx == C.Unknown

    @given(expressions())
    def test_bound_opposite_sign(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(-0.1, 0.1), pe.sin)
        assert cvx == C.Unknown

    @given(expressions())
    def test_positive_sin_linear_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(pi/2-0.1, pi/2+0.1), pe.sin)
        assert cvx == C.Concave

    @given(expressions())
    def test_positive_sin_convex_child_but_wrong_interval(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(pi/2-0.1, pi/2+0.1), pe.sin)
        assert cvx == C.Unknown

    @given(expressions())
    def test_positive_sin_convex_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(pi/2+0.1, pi-0.1), pe.sin)
        assert cvx == C.Concave

    @given(expressions())
    def test_positive_sin_concave_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(0.1, pi/2-0.1), pe.sin)
        assert cvx == C.Concave

    @given(expressions())
    def test_negative_sin_linear_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(1.5*pi-0.1, 1.5*pi+0.1), pe.sin)
        assert cvx == C.Convex

    @given(expressions())
    def test_negative_sin_concave_child_but_wrong_interval(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(1.5*pi-0.1, 1.5*pi+0.1), pe.sin)
        assert cvx == C.Unknown

    @given(expressions())
    def test_negative_sin_concave_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Constant, I(pi+0.1, 1.5*pi-0.1), pe.sin)
        assert cvx == C.Convex

    @given(expressions())
    def test_negative_sin_convex_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(1.5*pi, 2*pi), pe.sin)
        assert cvx == C.Convex


class TestCos(UnaryFunctionTest):
    @given(expressions())
    def test_bound_size_too_big(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(-2, 2), pe.cos)
        assert cvx == C.Unknown

    @given(expressions())
    def test_bound_opposite_sign(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(pi/2-0.1, pi/2+0.1), pe.cos)
        assert cvx == C.Unknown

    @given(expressions())
    def test_positive_cos_linear_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(0, 0.5*pi), pe.cos)
        assert cvx == C.Concave

    @given(expressions())
    def test_positive_cos_convex_child_but_wrong_interval(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(0, 0.5*pi), pe.cos)
        assert cvx == C.Unknown

    @given(expressions())
    def test_positive_cos_convex_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(1.5*pi, 2*pi), pe.cos)
        assert cvx == C.Concave

    @given(expressions())
    def test_positive_cos_concave_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(0, 0.5*pi), pe.cos)
        assert cvx == C.Concave

    @given(expressions())
    def test_negative_cos_linear_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(0.6*pi, 1.4*pi), pe.cos)
        assert cvx == C.Convex

    @given(expressions())
    def test_negative_cos_concave_child_but_wrong_interval(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(0.6*pi, 0.9*pi), pe.cos)
        assert cvx == C.Unknown

    @given(expressions())
    def test_negative_cos_concave_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(1.1*pi, 1.4*pi), pe.cos)
        assert cvx == C.Convex

    @given(expressions())
    def test_negative_cos_convex_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(0.6*pi, 0.9*pi), pe.cos)
        assert cvx == C.Convex



class TestTan(UnaryFunctionTest):
    @given(expressions())
    def test_bound_size_too_big(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(-2, 2), pe.tan)
        assert cvx == C.Unknown

    @given(expressions())
    def test_bound_opposite_sign(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Linear, M.Constant, I(-0.1, 0.1), pe.tan)
        assert cvx == C.Unknown

    @given(expressions())
    def test_positive_tan_convex_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(pi, 1.5*pi), pe.tan)
        assert cvx == C.Convex

    @given(expressions())
    def test_positive_tan_concave_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(pi, 1.5*pi), pe.tan)
        assert cvx == C.Unknown

    @given(expressions())
    def test_negative_tan_convex_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(-1.5*pi, -pi), pe.tan)
        assert cvx == C.Unknown

    @given(expressions())
    def test_negative_tan_concave_child(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(-0.49*pi, -0.1), pe.tan)
        assert cvx == C.Concave


class TestAsin(UnaryFunctionTest):
    @given(expressions())
    def test_is_concave(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(-1.0, 0), pe.asin)
        assert cvx == C.Concave

    @pytest.mark.parametrize('bounds', [I(-1, 0.1), I(-1.1, 0.0), I(None, None)])
    @given(child=expressions())
    def test_is_not_concave(self, visitor, child, bounds):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, bounds, pe.asin)
        assert cvx == C.Unknown

    @given(child=expressions())
    def test_is_convex(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(0, 1), pe.asin)
        assert cvx == C.Convex

    @pytest.mark.parametrize('bounds', [I(-0.1, 1), I(0.0, 1.1), I(None, None)])
    @given(child=expressions())
    def test_is_not_convex(self, visitor, child, bounds):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, bounds, pe.asin)
        assert cvx == C.Unknown


class TestAcos(UnaryFunctionTest):
    @given(child=expressions())
    def test_is_concave(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(0, 1), pe.acos)
        assert cvx == C.Concave

    @pytest.mark.parametrize('bounds', [I(-0.1, 1), I(0.0, 1.1), I(None, None)])
    @given(child=expressions())
    def test_is_not_concave(self, visitor, child, bounds):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, bounds, pe.acos)
        assert cvx == C.Unknown

    @given(child=expressions())
    def test_is_convex(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(-1, 0), pe.acos)
        assert cvx == C.Convex

    @pytest.mark.parametrize('bounds', [I(-1, 0.1), I(-1.1, 0.0), I(None, None)])
    @given(child=expressions())
    def test_is_not_convex(self, visitor, child, bounds):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, bounds, pe.acos)
        assert cvx == C.Unknown


class TestAtan(UnaryFunctionTest):
    @given(child=expressions())
    def test_is_concave(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, I(0, None), pe.atan)
        assert cvx == C.Concave

    @pytest.mark.parametrize('bounds', [I(None, 0), I(None, None)])
    @given(child=expressions())
    def test_is_not_concave(self, visitor, child, bounds):
        cvx = self._rule_result(visitor, child, C.Concave, M.Unknown, bounds, pe.atan)
        assert cvx == C.Unknown

    @given(child=expressions())
    def test_is_convex(self, visitor, child):
        cvx = self._rule_result(visitor, child, C.Convex, M.Unknown, I(None, 0), pe.atan)
        assert cvx == C.Convex

    @pytest.mark.parametrize('bounds', [I(0, None), I(None, None)])
    @given(child=expressions())
    def test_is_not_convex(self, visitor, child, bounds):
        cvx = self._rule_result(visitor, child, C.Convex, C.Unknown, bounds, pe.atan)
        assert cvx == C.Unknown


class TestPowConstantBase:
    def _rule_result(self, visitor, base, expo, cvx_expo, mono_expo, bounds_expo):
        convexity = ComponentMap()
        convexity[expo] = cvx_expo
        convexity[base] = C.Linear
        mono = ComponentMap()
        mono[expo] = mono_expo
        mono[base] = M.Constant
        bounds = ComponentMap()
        bounds[base] = I(base, base)
        bounds[expo] = bounds_expo

        expr = PowExpression([base, expo])
        matched, result = visitor.visit_expression(expr, convexity, mono, bounds)
        assert matched
        return result

    @pytest.mark.parametrize(
        'cvx_expo,mono_expo,bounds_expo',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(
        base=reals(max_value=-0.01, allow_infinity=False),
        expo=expressions(),
    )
    def test_negative_base(self, visitor, base, expo, cvx_expo, mono_expo, bounds_expo):
        cvx = self._rule_result(visitor, base, expo, cvx_expo, mono_expo, bounds_expo)
        assert cvx == C.Unknown

    @pytest.mark.parametrize('cvx_expo,mono_expo,bounds_expo',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(
        base=reals(min_value=0.001, max_value=0.999),
        expo=expressions(),
    )
    def test_base_between_0_and_1(self, visitor, base, expo, cvx_expo, mono_expo, bounds_expo):
        if cvx_expo == C.Concave or cvx_expo == C.Linear:
            expected = C.Convex
        else:
            expected = C.Unknown
        cvx = self._rule_result(visitor, base, expo, cvx_expo, mono_expo, bounds_expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_expo,mono_expo,bounds_expo',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(
        base=reals(min_value=1, allow_infinity=False),
        expo=expressions(),
    )
    def test_base_gt_1(self, visitor, base, expo, cvx_expo, mono_expo, bounds_expo):
        if cvx_expo == C.Convex or cvx_expo == C.Linear:
            expected = C.Convex
        else:
            expected = C.Unknown
        cvx = self._rule_result(visitor, base, expo, cvx_expo, mono_expo, bounds_expo)
        assert cvx == expected


class TestPowConstantExponent(object):
    def _rule_result(self, visitor, base, cvx_base, mono_base, bounds_base, expo):
        convexity = ComponentMap()
        convexity[base] = cvx_base
        convexity[expo] = C.Linear
        mono = ComponentMap()
        mono[base] = mono_base
        mono[expo] = M.Constant
        bounds = ComponentMap()
        bounds[base] = bounds_base
        bounds[expo] = I(expo, expo)
        expr = PowExpression([base, expo])
        matched, result = visitor.visit_expression(expr, convexity, mono, bounds)
        assert matched
        return result

    @pytest.mark.parametrize(
        'cvx_base,mono_base,bounds_base',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(base=expressions())
    def test_exponent_equals_0(self, visitor, base, cvx_base, mono_base, bounds_base):
        cvx = self._rule_result(visitor, base, cvx_base, mono_base, bounds_base, 0.0)
        assert cvx == C.Linear

    @pytest.mark.parametrize(
        'cvx_base,mono_base,bounds_base',
        itertools.product(
            [C.Convex, C.Concave, C.Linear, C.Unknown],
            [M.Nondecreasing, M.Nonincreasing, M.Unknown],
            [I(None, 0), I(0, None), I(None, None)],
        )
    )
    @given(base=expressions())
    def test_exponent_equals_1(self, visitor, base, cvx_base, mono_base, bounds_base):
        cvx = self._rule_result(visitor, base, cvx_base, mono_base, bounds_base, 1.0)
        assert cvx == cvx_base

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Linear, M.Nondecreasing, I(None, None), C.Convex),

        (C.Convex, M.Unknown, I(0, None), C.Convex),
        (C.Convex, M.Unknown, I(None, 0), C.Unknown),

        (C.Concave, M.Unknown, I(0, None), C.Unknown),
        (C.Concave, M.Unknown, I(None, 0), C.Convex),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000),
    )
    def test_positive_even_integer(self, visitor, base, expo, cvx_base, mono_base,
                                   bounds_base, expected):
        cvx = self._rule_result(visitor, base, cvx_base, mono_base, bounds_base, 2*expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Convex, M.Unknown, I(None, 0), C.Convex),
        (C.Convex, M.Unknown, I(0, None), C.Concave),

        (C.Concave, M.Unknown, I(0, None), C.Convex),
        (C.Concave, M.Unknown, I(None, 0), C.Concave),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000),
    )
    def test_negative_even_integer(self, visitor, base, expo, cvx_base, mono_base,
                                   bounds_base, expected):
        cvx = self._rule_result(visitor, base, cvx_base, mono_base, bounds_base, -2*expo)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Convex, M.Unknown, I(0, None), C.Convex),
        (C.Convex, M.Unknown, I(None, 0), C.Unknown),
        (C.Concave, M.Unknown, I(None, 0), C.Concave),
        (C.Concave, M.Unknown, I(0, None), C.Unknown),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000),
    )
    def test_positive_odd_integer(self, visitor, base, expo, cvx_base, mono_base,
                                  bounds_base, expected):
        cvx = self._rule_result(visitor, base, cvx_base, mono_base, bounds_base, 2*expo+1)
        assert cvx == expected

    @pytest.mark.parametrize('cvx_base,mono_base,bounds_base,expected', [
        (C.Concave, M.Unknown, I(0, None), C.Convex),
        (C.Concave, M.Unknown, I(None, 0), C.Unknown),
        (C.Convex, M.Unknown, I(None, 0), C.Concave),
        (C.Convex, M.Unknown, I(0, None), C.Unknown),
    ])
    @given(
        base=expressions(),
        expo=st.integers(min_value=1, max_value=1000),
    )
    def test_negative_odd_integer(self, visitor, base, expo, cvx_base, mono_base,
                                  bounds_base, expected):
        cvx = self._rule_result(visitor, base, cvx_base, mono_base, bounds_base, -2*expo+1)
        assert cvx == expected

    @given(
        base=expressions(),
        expo=reals(min_value=1, allow_infinity=False, max_value=1000),
    )
    def test_positive_gt_1_non_integer_negative_base(self, visitor, base, expo):
        expo = expo + 1e-3
        assume(expo != int(expo))
        cvx = self._rule_result(visitor, base, C.Convex, M.Unknown, I(None, -1), expo)
        assert cvx == C.Unknown

    @given(
        base=expressions(),
        expo=reals(min_value=1, allow_infinity=False, max_value=1000),
    )
    def test_positive_gt_1_non_integer(self, visitor, base, expo):
        expo = expo + 1e-5 # make it positive
        assume(expo != int(expo))
        cvx = self._rule_result(visitor, base, C.Convex, M.Unknown, I(0, None), expo)
        assert cvx == C.Convex

    @pytest.mark.parametrize('cvx,expected', [(C.Convex, C.Concave), (C.Concave, C.Convex)])
    @given(
        base=expressions(),
        expo=reals(min_value=-1000, max_value=0, allow_infinity=False),
    )
    def test_positive_lt_0_non_integer(self, visitor, base, expo, cvx, expected):
        expo = expo - 1e-5 # make it negative
        assume(not almosteq(expo, int(expo)))
        cvx = self._rule_result(visitor, base, cvx, M.Unknown, I(0, None), expo)
        assert cvx == expected

    @given(
        base=expressions(),
        expo=reals(min_value=0, max_value=1, allow_infinity=False),
    )
    def test_positive_0_1_non_integer(self, visitor, base, expo):
        assume(not almosteq(expo, int(expo)))
        cvx = self._rule_result(visitor, base, C.Concave, M.Unknown, I(0, None), expo)
        assert cvx == C.Concave


@pytest.mark.skip('Pyomo has no quadratic')
class TestQuadratic:
    def _rule_result(self, A):
        n = A.shape[0]
        var = [PE(ET.Variable) for _ in range(n)]
        terms = []
        for i in range(n):
            for j in range(i+1):
                terms.append(BilinearTerm(var[i], var[j], A[i, j]))

        expr = PE(ET.Quadratic, terms=terms, children=var)
        rule = QuadraticRule()
        return rule.apply(expr, None)

    @given(coefs=st.lists(reals(min_value=0, allow_infinity=False), min_size=1))
    def test_sum_of_squares_is_convex_with_positive_coefficients(self, coefs):
        assume(any([c > 0 and not np.isclose(c, 0) for c in coefs]))
        A = np.eye(len(coefs)) * coefs
        cvx = self._rule_result(A)
        assert cvx == C.Convex

    @given(coefs=st.lists(reals(max_value=0, allow_infinity=False), min_size=1))
    def test_sum_of_squares_is_concave_with_negative_coefficients(self, coefs):
        assume(any([c < 0 and not np.isclose(c, 0) for c in coefs]))
        A = np.eye(len(coefs)) * coefs
        cvx = self._rule_result(A)
        assert cvx == C.Concave

    @given(
        neg_coefs=st.lists(reals(max_value=0.0, allow_infinity=False), min_size=1),
        pos_coefs=st.lists(reals(min_value=0.0, allow_infinity=False), min_size=1))
    def test_sum_of_squares_is_unknown_otherwise(self, neg_coefs, pos_coefs):
        assume(any([n < 0 and not np.isclose(n, 0) for n in neg_coefs]))
        assume(any([p > 0 and not np.isclose(p, 0) for p in pos_coefs]))
        coefs = neg_coefs + pos_coefs
        A = np.eye(len(coefs)) * coefs
        cvx = self._rule_result(A)
        assert cvx == C.Unknown

    @given(st.integers(min_value=1, max_value=100))
    def test_positive_definite_is_convex(self, n):
        B = np.random.randn(n, n)
        A = np.eye(n) * n + 0.5 * (B + B.T)
        cvx = self._rule_result(A)
        assert cvx == C.Convex

    @given(st.integers(min_value=1, max_value=100))
    def test_negative_definite_is_concave(self, n):
        B = np.random.randn(n, n)
        A = -np.eye(n) * n - 0.5 * (B + B.T)
        cvx = self._rule_result(A)
        assert cvx == C.Concave

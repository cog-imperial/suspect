import hypothesis.strategies as st
import numpy as np
import pyomo.environ as pe
import suspect.pyomo.expressions as py5
from suspect.interval import Interval


@st.composite
def coefficients(draw, min_value=None, max_value=None):
    return draw(st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=False,
    ))


@st.composite
def intervals(draw, allow_infinity=True):
    a = draw(reals(allow_infinity=allow_infinity))
    if np.isinf(a):
        allow_infinity=False
    b = draw(reals(allow_infinity=allow_infinity))
    lb, ub = min(a, b), max(a, b)
    return Interval(lb, ub)


@st.composite
def reals(draw, min_value=None, max_value=None, allow_infinity=True):
    if min_value is not None and max_value is not None:
        allow_infinity = False
    return draw(st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=allow_infinity
    ))


domains = lambda: st.one_of(
    st.just(pe.Reals),
    st.just(pe.Integers),
    st.just(pe.PositiveIntegers),
    st.just(pe.Binary))


@st.composite
def variables(draw, domains=domains()):
    domain = draw(domains)
    return pe.Var(domain=domain)


constants = lambda: st.floats(min_value=-1e19, max_value=1e19, allow_nan=False, allow_infinity=False)


leaves = lambda: st.one_of(variables(), constants())


unary_function_types = st.one_of(
    st.just('cos'),
    st.just('sin'),
    st.just('tan'),
    st.just('acos'),
    st.just('asin'),
    st.just('atan'),
    st.just('exp'),
    st.just('log'),
    st.just('abs'))


@st.composite
def unary_functions(draw, child):
    ch = draw(child)
    name = draw(unary_function_types)
    if name == 'abs':
        return abs(ch)
    return getattr(pe, name)(ch)


@st.composite
def monomials(draw, vars=variables()):
    var = draw(vars)
    const = draw(constants())
    return py5.MonomialTermExpression([const, var])


@st.composite
def products(draw, arg1, arg2):
    return py5.ProductExpression([draw(arg1), draw(arg2)])


@st.composite
def sums(draw, args):
    children = draw(args)
    return py5.SumExpression(children)


def _combine_nodes(nodes):
    return st.one_of(
        unary_functions(nodes),
        products(nodes, nodes),
        sums(st.lists(nodes, min_size=2)))


def expressions(variables=variables(), max_leaves=100):
    return st.recursive(variables | monomials(variables), _combine_nodes)


@st.composite
def constraints(draw, variables=variables()):
    lb = draw(constants() | st.none())
    if lb is None:
        ub = draw(constants())
    else:
        ub = draw(constants() | st.none())

    if lb is not None and ub is not None:
        lb, ub = min(lb, ub), max(lb, ub)

    root_expr = draw(expressions(variables))

    if lb is not None:
        lb = py5.NumericConstant(lb)
    if ub is not None:
        ub = py5.NumericConstant(ub)

    expr = pe.inequality(lb, root_expr, ub)
    return pe.Constraint(expr=expr)


@st.composite
def models(draw, max_constraints=None):
    m = pe.ConcreteModel()

    vars = draw(st.lists(variables(), min_size=1))
    for i, var in enumerate(vars):
        setattr(m, 'x_{}'.format(i), var)

    one_of_vars = st.one_of([st.just(v) for v in vars])

    m.obj = pe.Objective(expr=draw(expressions(one_of_vars)))

    cons = draw(st.lists(constraints(one_of_vars), max_size=max_constraints))
    for i, con in enumerate(cons):
        setattr(m, 'c_{}'.format(i), con)

    return m

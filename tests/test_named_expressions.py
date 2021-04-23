import pyomo.environ as pe
import suspect
from suspect.monotonicity.monotonicity import Monotonicity
import math
from pyomo.common.collections import ComponentSet
from pyomo.core.expr.visitor import identify_variables


def create_model():
    m = pe.ConcreteModel()
    m.x = pe.Var(bounds=(-1, 1))
    m.y = pe.Var()
    m.z = pe.Var()
    m.p = pe.Param(mutable=True)

    m.e = pe.Expression(expr=m.x - 1)
    m.obj = pe.Objective(expr=m.x**2 + m.y**2 + m.z**2)
    m.c1 = pe.Constraint(expr=(0, m.y - pe.exp(m.e), None))
    m.c2 = pe.Constraint(expr=(0, m.z - pe.exp(m.e), None))

    return m


def test_connected_model():
    m = create_model()
    cm, cm_m_map = suspect.create_connected_model(m)
    assert cm.c1.body.args[1].args[0].args[0] is cm.e
    assert cm.c2.body.args[1].args[0].args[0] is cm.e
    assert cm.c1.body.args[1].args[0] is cm.c2.body.args[1].args[0]
    assert cm.e is not m.e
    cm_copy = cm.clone()
    assert cm.x not in ComponentSet(identify_variables(cm_copy.c1.body))
    assert cm_copy.x in ComponentSet(identify_variables(cm_copy.c1.body))


def test_fbbt():
    m = create_model()
    cm, cm_m_map = suspect.create_connected_model(m)
    bounds = suspect.perform_fbbt(cm)
    assert math.isclose(bounds[cm.e].lower_bound, -2)
    assert math.isclose(bounds[cm.e].upper_bound, 0)


def test_monotonicity_and_convexity():
    m = create_model()
    cm, cm_m_map = suspect.create_connected_model(m)
    bounds = suspect.perform_fbbt(cm)
    mono, cvx = suspect.propagate_special_structure(cm, bounds=bounds)
    assert mono[cm.e] == Monotonicity.Nondecreasing

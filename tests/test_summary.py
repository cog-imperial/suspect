# pylint: skip-file
import pytest
import pyomo.environ as aml
from suspect import detect_special_structure
from suspect.convexity import Convexity


@pytest.fixture
def problem():
    m = aml.ConcreteModel()
    m.x = aml.Var(bounds=(-2.0, 2.0))
    m.y = aml.Var(bounds=(0.01, 1.5))

    m.obj = aml.Objective(expr=m.x + m.y)
    m.c0 = aml.Constraint(expr=m.x**2 + m.y ** 2 <= 1.0)
    m.c1 = aml.Constraint(expr=aml.sin(m.y) >= 0.0)

    return m


def test_summary(problem):
    info = detect_special_structure(problem)
    assert info.num_variables() == 2
    assert info.num_integers() == 0
    assert info.num_binaries() == 0
    assert info.num_constraints() == 2
    conscurvature =  info.conscurvature()
    assert conscurvature['c0'] == Convexity.Convex
    assert conscurvature['c1'] == Convexity.Convex

    objcurvature = info.objcurvature()
    assert objcurvature['obj'] == Convexity.Linear

    objtype = info.objtype()
    assert objtype['obj'] == 'linear'

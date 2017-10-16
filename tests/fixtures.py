import pytest
import pyomo.environ as aml


@pytest.fixture
def model():
    m = aml.ConcreteModel()
    m.I = range(10)
    m.J = range(5)
    m.x = aml.Var(m.I)
    m.y = aml.Var(m.I, m.J, domain=aml.NonNegativeReals, bounds=(0, 1))
    m.z = aml.Var(m.I, bounds=(0, 10))
    return m

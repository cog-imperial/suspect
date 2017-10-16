import pytest
import pyomo.environ as aml
from convexity_detection.util import *
from fixtures import model


def test_model_variables(model):
    assert 10 + 50 + 10 == len([_ for _ in model_variables(model)])


def test_model_constraints(model):
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 0
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum(m.z[i] for i in m.I) <= 5
    )
    model.cons3 = aml.Constraint(
        model.I,
        rule=lambda m, i: -1 <= m.x[i] <= 1
    )
    model.cons4 = aml.Constraint(
        model.I, model.J,
        rule=lambda m, i, j: -20 <= m.x[i] * m.y[i, j] <= 20,
    )
    assert 1 + 1 + 10 + 50 == len([_ for _ in model_constraints(model)])

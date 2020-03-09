# pylint: skip-file
import pytest
from pathlib import Path
from suspect.pyomo.util import (
    model_variables,
    model_objectives,
    model_constraints,
)
from suspect.pyomo import read_osil
import pyomo.environ as pe
from pyomo.core.expr.sympy_tools import sympyify_expression


current_dir = Path(__file__).resolve().parent


def test_osil_read():
    filename = current_dir / 'osil' / 'example1.xml'
    problem = read_osil(str(filename))
    variables = list(model_variables(problem))
    constraints = list(model_constraints(problem))
    objectives = list(model_objectives(problem))
    assert len(variables) == 35
    assert len(constraints) == 18
    assert len(objectives) == 1

    m = problem

    e9_body_expected = m.x33 - (m.x32*pe.log(m.x22)/(1.0-m.x32))
    e9_body_actual = m.e9.body
    assert sympyify_expression(e9_body_expected - e9_body_actual)[1] == 0

    e11_body_expected = m.x35 - ((m.x33 - m.x34*(m.x32+1.0))/(m.x32+3.0)/(1.0-m.x32)**2/m.x34)
    e11_body_actual = m.e11.body
    assert sympyify_expression(e11_body_expected - e11_body_actual)[1] == 0

    e14_body_expected = m.x26 - m.x22 * m.x28
    e14_body_actual = m.e14.body
    assert sympyify_expression(e14_body_expected - e14_body_actual)[1] == 0

    e15_body_expected = m.x27 - m.x22 * m.x29
    e15_body_actual = m.e15.body
    assert sympyify_expression(e15_body_expected - e15_body_actual)[1] == 0

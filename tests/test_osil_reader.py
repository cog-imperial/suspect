# pylint: skip-file
import pytest
from pathlib import Path
from suspect.pyomo.util import (
    model_variables,
    model_objectives,
    model_constraints,
)
from suspect.pyomo import read_osil


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

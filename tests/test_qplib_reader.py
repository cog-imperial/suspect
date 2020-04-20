import pytest
import requests
import pyomo.environ as pe
import contextlib
import tempfile
from suspect.pyomo import read_qplib, create_connected_model


def _download_qplib_file(file, problem_name):
    url = 'http://qplib.zib.de/qplib/QPLIB_{}.qplib'.format(problem_name)
    response = requests.get(url, stream=True)
    file.write(response.content)


@contextlib.contextmanager
def qplib_problem(problem_name):
    with tempfile.NamedTemporaryFile(suffix='.qplib') as f:
        _download_qplib_file(f, problem_name)
        f.seek(0)
        yield f


def read_qplib_problem(filename):
    result = read_qplib(filename)
    assert isinstance(result, pe.ConcreteModel)


@pytest.mark.parametrize('problem_name', ['0018', '0681', '0685', '3554'])
def test_read_qplib_problem(benchmark, problem_name):
    with qplib_problem(problem_name) as f:
        benchmark(read_qplib_problem, f.name)


@pytest.mark.parametrize('problem_name', ['0018', '0681', '0685', '3554'])
def test_convert_qplib_problem(benchmark, problem_name):
    with qplib_problem(problem_name) as f:
        model = read_qplib(f.name)
        connected_model, _ = benchmark(create_connected_model, model)
        assert isinstance(connected_model, pe.ConcreteModel)


def test_profile():
    with qplib_problem('3554') as f:
        model = read_qplib(f.name)
        connected_model, _ = create_connected_model(model)
        assert isinstance(connected_model, pe.ConcreteModel)

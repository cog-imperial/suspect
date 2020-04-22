import pytest
import pyomo.environ as pe
from suspect.pyomo import create_connected_model
from suspect.fbbt import perform_fbbt
from suspect.pyomo.quadratic import enable_standard_repn_for_quadratic_expression


enable_standard_repn_for_quadratic_expression()


def model_with_n_variables(n):
    m = pe.ConcreteModel()
    m.I = range(n)
    m.x = pe.Var(m.I, domain=pe.Reals)
    m.y = pe.Var(m.I, domain=pe.Integers, bounds=(1, 10))

    m.obj = pe.Objective(expr=sum(m.x[i]**2.0 for i in m.I))

    num_quad_cons = 10
    @m.Constraint(range(num_quad_cons))
    def quadratic_cons(m, k):
        num_vars = n // num_quad_cons
        return sum(
            m.x[i] * m.x[j]
            for i in range(k*num_vars, (k+1)*num_vars)
            for j in range(k*num_vars, i)
        ) <= 2*n

    @m.Constraint(m.I)
    def bilinear_cons1(m, i):
        return m.x[i]*m.y[i] >= 0

    @m.Constraint(m.I)
    def bilinear_cons2(m, i):
        return m.x[i]*m.y[i] <= m.y[i]

    cm, _ = create_connected_model(m)
    return cm


@pytest.fixture
def small_model():
    return model_with_n_variables(20)


@pytest.fixture
def medium_model():
    return model_with_n_variables(100)


@pytest.fixture
def large_model():
    return model_with_n_variables(300)


def test_initial_fbbt_performance_on_small_model(benchmark, small_model):
    objective_bounds = pe.ComponentMap()
    objective_bounds[small_model.obj.expr] = (None, 100.0)
    result = benchmark(perform_fbbt, small_model, **{'objective_bounds': objective_bounds})


def test_initial_fbbt_performance_on_medium_model(benchmark, medium_model):
    objective_bounds = pe.ComponentMap()
    objective_bounds[medium_model.obj.expr] = (None, 100.0)
    result = benchmark(perform_fbbt, medium_model, **{'objective_bounds': objective_bounds})


@pytest.mark.skip()
def test_initial_fbbt_performance_on_large_model(benchmark, large_model):
    objective_bounds = pe.ComponentMap()
    objective_bounds[large_model.obj.expr] = (None, 100.0)
    result = benchmark(perform_fbbt, large_model, **{'objective_bounds': objective_bounds})


def test_recompute_fbbt_performance_on_small_model(benchmark, small_model):
    objective_bounds = pe.ComponentMap()
    objective_bounds[small_model.obj.expr] = (None, 100.0)
    bounds = perform_fbbt(small_model, objective_bounds=objective_bounds)
    kwargs = {'objective_bounds': objective_bounds, 'initial_bounds': bounds}
    result = benchmark(perform_fbbt, small_model, **kwargs)


def test_recompute_fbbt_performance_on_medium_model(benchmark, medium_model):
    objective_bounds = pe.ComponentMap()
    objective_bounds[medium_model.obj.expr] = (None, 100.0)
    bounds = perform_fbbt(medium_model, objective_bounds=objective_bounds)
    kwargs = {'objective_bounds': objective_bounds, 'initial_bounds': bounds}
    result = benchmark(perform_fbbt, medium_model, **kwargs)


@pytest.mark.skip()
def test_recompute_fbbt_performance_on_large_model(benchmark, large_model):
    objective_bounds = pe.ComponentMap()
    objective_bounds[large_model.obj.expr] = (None, 100.0)
    bounds = perform_fbbt(large_model, objective_bounds=objective_bounds)
    kwargs = {'objective_bounds': objective_bounds, 'initial_bounds': bounds}
    result = benchmark(perform_fbbt, large_model, **kwargs)

# pylint: skip-file
import pytest
import pyomo.environ as pe
from suspect.pyomo import dag_from_pyomo_model
from suspect.dag.tags import LinearExpressionTag


class TestLinearExpressionTag:
    def test_sum_of_monomials(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(range(5))

        m.c = pe.Constraint(expr=2.0 + sum((i + 2)*m.x[i] for i in range(5)) + m.x[0] + 3.0 * m.x[1] + 3.0 >= 0)
        dag = dag_from_pyomo_model(m)
        body = dag.constraints['c'].body
        tag = LinearExpressionTag.from_expr(dag, body)
        x0 = dag.variables['x[0]']
        x1 = dag.variables['x[1]']
        assert tag is not None
        assert tag.constant == 5.0
        assert tag.coefficient(x0) == 3.0
        assert tag.coefficient(x1) == 6.0

    def test_sum_of_variables(self):
        m = pe.ConcreteModel()
        m.x = pe.Var(range(5))

        m.c = pe.Constraint(expr=sum(m.x[i] for i in range(5)) >= 0)
        dag = dag_from_pyomo_model(m)
        body = dag.constraints['c'].body
        tag = LinearExpressionTag.from_expr(dag, body)
        assert tag is not None
        for var in dag.variables.values():
            assert tag.coefficient(var) == 1.0

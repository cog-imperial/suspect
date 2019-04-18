# pylint: skip-file
import pytest
import pyomo.environ as aml
from tests.conftest import PlaceholderExpression
from suspect.pyomo import dag_from_pyomo_model
from suspect.dag.dag import ProblemDag
from suspect.dag.iterator import DagForwardIterator, DagBackwardIterator
import suspect.pyomo.expressions as pex
from hypothesis import given, assume
import hypothesis.strategies as st


def create_problem():

    # build a dag that looks like this:
    #
    #            c0    c1
    #            |     |
    #            |     |
    #            +    |.|
    #            | \   |
    #            |   \ |
    #            S     ^
    #          /  \   / \
    #        x0     x1   c

    m = aml.ConcreteModel()

    m.x0 = aml.Var()
    m.x1 = aml.Var()

    m.c0 = aml.Constraint(expr=m.x0 - m.x1 + m.x1 ** 3 <= 0)
    m.c1 = aml.Constraint(expr=abs(m.x1 ** 3) <= 10.0)

    return dag_from_pyomo_model(m)


@pytest.fixture
def problem_dag():
    return create_problem()


class FakeForwardVisitor(object):
    def __init__(self, visit_returns):
        self.seen = set()
        self._visit_returns = visit_returns

    def visit(self, expr, ctx):
        self.seen.add(id(expr))
        return self._visit_returns(expr)


class TestDagForwardIterator(object):
    def test_no_starting_point(self, problem_dag):
        it = DagForwardIterator()
        visitor = FakeForwardVisitor(lambda v: True)
        changes = it.iterate(problem_dag, visitor, None)
        assert changes[0].is_variable_type()
        assert changes[-1].is_sink
        assert len(changes) == 10

    def test_early_stop(self, problem_dag):
        it = DagForwardIterator()
        visitor = FakeForwardVisitor(
            lambda v: False if isinstance(v, pex.PowExpression) else True,
        )
        changes = it.iterate(problem_dag, visitor, None)
        assert len(changes) == 10-3

    def test_starting_point(self, problem_dag):
        it = DagForwardIterator()
        visitor = FakeForwardVisitor(lambda v: False)
        changes = it.iterate(problem_dag, visitor, None, starting_vertices=[])
        assert len(visitor.seen) == 4
        assert len(changes) == 0


class FakeBackwardVisitor(object):
    def __init__(self, visit_returns):
        self.seen = {}
        self._visit_returns = visit_returns

    def visit(self, expr, ctx):
        self.seen[id(expr)] = expr
        return self._visit_returns(expr)


class TestDagBackwardIterator(object):
    def test_no_starting_point(self, problem_dag):
        it = DagBackwardIterator()
        visitor = FakeBackwardVisitor(lambda v: True)
        changes = it.iterate(problem_dag, visitor, None)
        assert len(changes) == 10
        assert changes[0].is_sink
        assert changes[-1].is_variable_type()

    def test_early_stop(self, problem_dag):
        it = DagBackwardIterator()
        visitor = FakeBackwardVisitor(
            lambda v: False if isinstance(v, pex.SumExpression) else True,
        )
        changes = it.iterate(problem_dag, visitor, None)
        assert len(changes) == 10-4

    def test_starting_point(self, problem_dag):
        it = DagBackwardIterator()
        visitor = FakeBackwardVisitor(lambda v: False)
        changes = it.iterate(problem_dag, visitor, None, starting_vertices=[])
        assert len(visitor.seen) == 2
        assert len(changes) == 0

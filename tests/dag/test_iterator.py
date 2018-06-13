# pylint: skip-file
import pytest
from tests.conftest import PlaceholderExpression
from suspect.dag.dag import ProblemDag
from suspect.dag.iterator import DagForwardIterator, DagBackwardIterator
import suspect.dag.expressions as dex
from hypothesis import given, assume
import hypothesis.strategies as st


@pytest.fixture
def problem_dag():
    dag = ProblemDag()

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

    x0 = dex.Variable('x0', None, None)
    dag.add_variable(x0)
    x1 = dex.Variable('x1', None, None)
    dag.add_variable(x1)
    c = dex.Constant(1.0)
    dag.add_vertex(c)

    n0 = dex.LinearExpression([1.0, -1.0], [x0, x1])
    x0.add_parent(n0)
    x1.add_parent(n0)
    dag.add_vertex(n0)

    n1 = dex.PowExpression([x1, c])
    x1.add_parent(n1)
    c.add_parent(n1)
    dag.add_vertex(n1)

    n2 = dex.SumExpression([n0, n1])
    n0.add_parent(n2)
    n1.add_parent(n2)
    dag.add_vertex(n2)

    n3 = dex.AbsExpression([n1])
    n1.add_parent(n3)
    dag.add_vertex(n3)

    c0 = dex.Constraint('c0', None, None, [n2])
    n2.add_parent(c0)
    dag.add_constraint(c0)

    c1 = dex.Constraint('c1', None, None, [n3])
    n3.add_parent(c1)
    dag.add_constraint(c1)

    return dag


class FakeForwardVisitor(object):
    def __init__(self, visit_returns):
        self.seen = set()
        self._visit_returns = visit_returns

    def visit(self, expr, ctx):
        self.seen.add(expr)
        return self._visit_returns(expr)


class TestDagForwardIterator(object):
    def test_no_starting_point(self, problem_dag):
        it = DagForwardIterator()
        visitor = FakeForwardVisitor(lambda v: True)
        changes = it.iterate(problem_dag, visitor, None)
        assert changes[0].is_source
        assert changes[-1].is_sink
        assert len(changes) == 9

    def test_early_stop(self, problem_dag):
        it = DagForwardIterator()
        visitor = FakeForwardVisitor(
            lambda v: False if isinstance(v, dex.PowExpression) else True,
        )
        changes = it.iterate(problem_dag, visitor, None)
        assert len(changes) == 9-3

    def test_starting_point(self, problem_dag):
        it = DagForwardIterator()
        visitor = FakeForwardVisitor(lambda v: False)
        changes = it.iterate(problem_dag, visitor, None, starting_vertices=[])
        assert len(visitor.seen) == 3
        assert len(changes) == 0


class FakeBackwardVisitor(object):
    def __init__(self, visit_returns):
        self.seen = set()
        self._visit_returns = visit_returns

    def visit(self, expr, ctx):
        self.seen.add(expr)
        return self._visit_returns(expr)


class TestDagBackwardIterator(object):
    def test_no_starting_point(self, problem_dag):
        it = DagBackwardIterator()
        visitor = FakeBackwardVisitor(lambda v: True)
        changes = it.iterate(problem_dag, visitor, None)
        assert changes[0].is_sink
        assert changes[-1].is_source
        assert len(changes) == 9

    def test_early_stop(self, problem_dag):
        it = DagBackwardIterator()
        visitor = FakeBackwardVisitor(
            lambda v: False if isinstance(v, dex.SumExpression) else True,
        )
        changes = it.iterate(problem_dag, visitor, None)
        assert len(changes) == 9-3

    def test_starting_point(self, problem_dag):
        it = DagBackwardIterator()
        visitor = FakeBackwardVisitor(lambda v: False)
        changes = it.iterate(problem_dag, visitor, None, starting_vertices=[])
        assert len(visitor.seen) == 2
        assert len(changes) == 0

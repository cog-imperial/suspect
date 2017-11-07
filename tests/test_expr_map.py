import pytest
import pyomo.environ as aml
from convexity_detection.exp_map import *
from convexity_detection.float_hash import RoundFloatHasher
from fixtures import model


def test_expr_hash_linear_expr(model):
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2
    )
    model.cons3 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I[:2]) == 3
    )

    hash_cons1 = expr_hash(model.cons1.expr)
    hash_cons2 = expr_hash(model.cons2.expr)
    hash_cons3 = expr_hash(model.cons3.expr)

    assert hash_cons1 == hash_cons2
    # TODO: fix too common collisions
    # assert hash_cons1 != hash_cons3

    # we want hashing of linear expr be associative
    model.cons4 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I[::-1]) == 2
    )

    hash_cons4 = expr_hash(model.cons4.expr)
    assert hash_cons1 == hash_cons4


def test_expr_hash_product_expr(model):
    model.cons1 = aml.Constraint(
        model.I,
        rule=lambda m, i: 0 <= m.x[i] * m.y[i, 0] <= 10
    )

    hash_cons1 = expr_hash(model.cons1[0].expr)
    for i in model.I:
        for j in model.J:
            if i == 0 and j == 0:
                continue
            # Collisions could happen, so a failure here is expected
            # in some rare cases
            ex = -10 <= model.x[i] * model.y[i, j] <= 100
            assert hash_cons1 != expr_hash(ex)


def test_expr_hash_unary_expr(model):
    model.cons1 = aml.Constraint(
        model.I,
        rule=lambda m, i: aml.cos(m.x[i]) <= 0.5
    )

    hash_cons1 = expr_hash(model.cons1[2].expr)
    ex1 = aml.cos(model.x[2]) <= 0.5
    assert hash_cons1 == expr_hash(ex1)


def test_expr_hash_float_hasher(model):
    model.cons1 = aml.Constraint(
        rule=lambda m: sum(m.x[i] for i in m.I) == 2
    )
    model.cons2 = aml.Constraint(
        rule=lambda m: sum((i+1)*m.x[i] for i in m.I) == 2
    )

    h = RoundFloatHasher(2)

    assert expr_hash(model.cons1.expr) == expr_hash(model.cons2.expr)
    assert expr_hash(model.cons1.expr, h) != expr_hash(model.cons2.expr, h)
    assert expr_hash(model.cons1.expr) != expr_hash(model.cons2.expr, h)

    model.cons3 = aml.Constraint(
        model.I,
        rule=lambda m, i: m.x[i] == i
    )
    model.cons4 = aml.Constraint(
        model.I,
        rule=lambda m, i: m.x[i] == (i + 1)
    )

    assert expr_hash(model.cons3[0]) == expr_hash(model.cons4[0])
    assert expr_hash(model.cons3[0], h) != expr_hash(model.cons4[0], h)

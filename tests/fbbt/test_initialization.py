# pylint: skip-file
import pytest
from pyomo.core.kernel.component_map import ComponentMap
from tests.conftest import PlaceholderExpression
from suspect.interval import Interval
from suspect.pyomo.expressions import UnaryFunctionExpression
from suspect.fbbt.initialization import BoundsInitializationVisitor
from suspect.fbbt.initialization.rules import (
    SqrtRule,
    LogRule,
    AsinRule,
    AcosRule,
)


def _visit(func_name):
    p = PlaceholderExpression()
    visitor = BoundsInitializationVisitor()
    root = UnaryFunctionExpression(p, name=func_name)
    bounds = ComponentMap()
    assert visitor.visit(root, bounds)
    print(bounds._dict)
    return bounds[p]


def test_sqrt():
    assert _visit('sqrt').is_nonnegative()


def test_log():
    assert _visit('log').is_nonnegative()


def test_asin():
    assert _visit('asin') == Interval(-1, 1)


def test_acos():
    assert _visit('acos') == Interval(-1, 1)

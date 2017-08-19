import pytest
import io
import pyomo.environ as aml
from convexity_detection.dot import *
from convexity_detection.pyomo_compat import *
import numpy as np
import pyomo.core.base.expr_pyomo4 as omo


set_pyomo4_expression_tree()


def test_dot():
    x = aml.Var()
    y = aml.Var()
    z = aml.Var()

    # e0 = 2*x*(3*y + z)
    with omo.bypass_clone_check():
        p = x + 1
        e0 = p * aml.log(p)
    # e0 = 7*(aml.sin(x/y) + x/y - aml.exp(y))*(x/y - aml.exp(y))

    f = io.StringIO()
    v = DotVisitor(f)
    v.visit(e0)
    print(e0)

    print(f.getvalue())

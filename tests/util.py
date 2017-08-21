from convexity_detection.pyomo_compat import set_pyomo4_expression_tree
import pyomo.environ as aml


set_pyomo4_expression_tree()


def _var(bounds=None):
    x = aml.Var(bounds=bounds)
    x.construct()
    return x

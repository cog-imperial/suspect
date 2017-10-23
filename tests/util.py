import pyomo.environ as aml


def _var(bounds=None):
    x = aml.Var(bounds=bounds)
    x.construct()
    return x

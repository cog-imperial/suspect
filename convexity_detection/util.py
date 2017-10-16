import pyomo.environ as aml


def model_variables(model):
    """Return a list of variables in the model"""
    for variables in model.component_map(aml.Var, active=True).itervalues():
        for idx in variables:
            yield variables[idx]


def model_constraints(model):
    """Return a list of constraints in the model"""
    for cons in model.component_map(aml.Constraint, active=True).itervalues():
        for idx in cons:
            yield cons[idx]

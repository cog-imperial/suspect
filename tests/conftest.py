from convexity_detection.pyomo_compat import set_pyomo4_expression_tree


def pytest_sessionstart(session):
    set_pyomo4_expression_tree()

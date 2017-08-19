from pyomo.core.base.expr import set_expression_tree_format
import pyomo.core.base.expr_common as common


def set_pyomo4_expression_tree():
    set_expression_tree_format(common.Mode.pyomo4_trees)

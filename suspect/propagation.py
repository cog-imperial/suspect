# Copyright 2017 Francesco Ceccon
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import pkg_resources
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.visitor import StreamBasedExpressionVisitor
from suspect.monotonicity import MonotonicityPropagationVisitor, propagate_expression_monotonicity
from suspect.convexity import ConvexityPropagationVisitor, propagate_expression_convexity
from suspect.dag.iterator import DagForwardIterator


logger = logging.getLogger('suspect')


def monotonicity_detection_entry_points():
    return pkg_resources.iter_entry_points('suspect.monotonicity_detection')


def convexity_detection_entry_points():
    return pkg_resources.iter_entry_points('suspect.convexity_detection')


class SpecialStructurePropagationVisitor(object):
    def __init__(self, problem):
        self._mono_visitors = [MonotonicityPropagationVisitor()]
        for entry_point in monotonicity_detection_entry_points():
            cls = entry_point.load()
            self._mono_visitors.append(cls(problem))

        logger.info(
            'Loaded %s monotonicity detectors: %s',
            len(self._mono_visitors), ', '.join([str(mono) for mono in self._mono_visitors])
        )

        self._cvx_visitors = [ConvexityPropagationVisitor()]
        for entry_point in convexity_detection_entry_points():
            cls = entry_point.load()
            self._cvx_visitors.append(cls(problem))

        logger.info(
            'Loaded %s convexity detectors: %s',
            len(self._cvx_visitors), ', '.join([str(cvx) for cvx in self._cvx_visitors])
        )

    def visit(self, expr, convexity, mono, bounds):
        for mono_visitor in self._mono_visitors:
            mono_known = mono_visitor.visit(expr, mono, bounds)
            if mono_known:
                break

        for cvx_visitor in self._cvx_visitors:
            cvx_known = cvx_visitor.visit(expr, convexity, mono, bounds)
            if cvx_known:
                break
        return [expr]


def propagate_special_structure(model, bounds, active=True):
    """Propagate special structure.

    Arguments
    ---------
    problem: ProblemDag
        the problem.
    bounds: dict-like
        a dict-like object containing bounds
    active : bool
        only propagate special structure on active components

    Returns
    -------
    monotonicity: dict-like
        monotonicity information for the problem.
    convexity: dict-like
        convexity information for the problem.
    """
    mono = ComponentMap()
    cvx = ComponentMap()

    for objective in model.component_data_objects(pyo.Objective, active=active, descend_into=True):
        _propagate_special_structure(objective.expr, bounds, cvx, mono)

    for constraint in model.component_data_objects(pyo.Constraint, active=active, descend_into=True):
        _propagate_special_structure(constraint.body, bounds, cvx, mono)

    return mono, cvx


def _propagate_special_structure(expr, bounds, cvx, mono):
    def enter_node(node):
        return None, None

    def exit_node(node, data):
        mono_result = propagate_expression_monotonicity(node, mono, bounds)
        mono[node] = mono_result
        result = propagate_expression_convexity(node, cvx, mono, bounds)
        cvx[node] = result

    return StreamBasedExpressionVisitor(
        enterNode=enter_node,
        exitNode=exit_node,
    ).walk_expression(expr)

from typing import NamedTuple

import pyomo.environ as pyo
from pyomo.core.expr.numeric_expr import (
    SumExpression,
)
from pyomo.repn.standard_repn import generate_standard_repn


def enable_standard_repn_for_quadratic_expression():
    from pyomo.repn.standard_repn import _repn_collectors, _collect_sum
    _repn_collectors[QuadraticExpression] = _collect_sum


class BilinearTerm(NamedTuple):
    var1: pyo.Var
    var2: pyo.Var
    coefficient: float


def _make_uid_index(v0, v1):
    v0_uid = id(v0)
    v1_uid = id(v1)
    if v0_uid < v1_uid:
        return (v0, v0_uid), (v1, v1_uid)
    return (v1, v1_uid), (v0, v0_uid)


class QuadraticExpression(SumExpression):
    def __init__(self, args):
        if isinstance(args, SumExpression):
            args = args.args
        if not isinstance(args, list):
            args = [args]
        super().__init__(args)
        self._coef_matrix = None

    @property
    def __class__(self):
        return SumExpression

    def coefficient(self, v0, v1):
        self._ensure_coef_matrix()
        (_, v0_uid), (_, v1_uid) = _make_uid_index(v0, v1)
        bilinear_term = self._coef_matrix.get((v0_uid, v1_uid), None)
        if bilinear_term is None:
            return 0.0
        return bilinear_term.coef

    @property
    def terms(self):
        self._ensure_coef_matrix()
        return self._coef_matrix.values()

    def _ensure_coef_matrix(self):
        if not hasattr(self, '_coef_matrix'):
            self._coef_matrix = None

        if self._coef_matrix is not None:
            return

        self._coef_matrix = dict()

        repn_result = generate_standard_repn(self, compute_values=False)
        assert not repn_result.linear_vars
        assert not repn_result.nonlinear_vars

        for ((v0, v1), coef) in zip(repn_result.quadratic_vars, repn_result.quadratic_coefs):
            (v0, v0_uid), (v1, v1_uid) = _make_uid_index(v0, v1)
            assert v0_uid <= v1_uid
            bilinear_term = BilinearTerm(v0, v1, coef)
            self._coef_matrix[(v0_uid, v1_uid)] = bilinear_term

        if repn_result.nonlinear_expr is not None:
            pass
        else:
            assert not repn_result.nonlinear_vars

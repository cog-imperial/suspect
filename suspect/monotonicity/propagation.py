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

import suspect.dag.expressions as dex
from suspect.dag.visitor import Dispatcher
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.monotonicity.product import product_monotonicity
from suspect.monotonicity.division import division_monotonicity
from suspect.monotonicity.linear import linear_monotonicity
from suspect.monotonicity.pow import pow_monotonicity


def propagate_monotonicity(dag, bounds):
    """Compute monotonicity of expressions in the problem.

    Arguments
    ---------
    dag: ProblemDag
      the problem
    bounds: dict-like
      bounds of the expressions.

    Returns
    -------
    monotonicity: dict-like
      monotonicity information for the problem
    """
    visitor = MonotonicityPropagationVisitor(bounds)
    dag.forward_visit(visitor)
    return visitor.result()


class MonotonicityPropagationVisitor(object):
    def __init__(self, bounds):
        self._bounds = bounds
        self._mono = {}
        self._dispatcher = Dispatcher(
            lookup={
                dex.Variable: self.visit_variable,
                dex.Constant: self.visit_constant,
                dex.Constraint: self.visit_constraint,
                dex.Objective: self.visit_objective,
                dex.ProductExpression: self.visit_product,
                dex.DivisionExpression: self.visit_division,
                dex.LinearExpression: self.visit_linear,
                dex.PowExpression: self.visit_pow,
                dex.SumExpression: self.visit_sum,

                dex.NegationExpression: self.visit_nonincreasing_function,
                dex.AbsExpression: self.visit_abs,
                dex.SqrtExpression: self.visit_nondecreasing_function,
                dex.ExpExpression: self.visit_nondecreasing_function,
                dex.LogExpression: self.visit_nondecreasing_function,
                dex.SinExpression: self.visit_sin,
                dex.CosExpression: self.visit_cos,
                dex.TanExpression: self.visit_nondecreasing_function,
                dex.AsinExpression: self.visit_nondecreasing_function,
                dex.AcosExpression: self.visit_nonincreasing_function,
                dex.AtanExpression: self.visit_nondecreasing_function,
            },
            allow_missing=False)

    def bound(self, expr):
        return self._bounds[id(expr)]

    def is_nonnegative(self, expr):
        return self.bound(expr).is_nonnegative()

    def is_nonpositive(self, expr):
        return self.bound(expr).is_nonpositive()

    def get(self, expr):
        return self._mono[id(expr)]

    def set(self, expr, mono):
        self._mono[id(expr)] = mono

    def result(self):
        return self._mono

    def __call__(self, expr):
        new_mono = self._dispatcher.dispatch(expr)
        if new_mono is None:
            raise RuntimeError('new_mono is None.')
        self.set(expr, new_mono)

    def visit_variable(self, _expr):
        return Monotonicity.Nondecreasing

    def visit_constant(self, _expr):
        return Monotonicity.Constant

    def visit_constraint(self, expr):
        if expr.bounded_below() and expr.bounded_above():
            # l <= g(x) <= u
            mono = self.get(expr.children[0])
            if mono.is_constant():
                return mono
            else:
                return Monotonicity.Unknown
        elif expr.bounded_below():
            # l <= g(x) => -g(x) <= -l
            mono = self.get(expr.children[0])
            return mono.negate()
        elif expr.bounded_above():
            # g(x) <= u
            return self.get(expr.children[0])
        else:
            raise RuntimeError('Constraint with no bounds')

    def visit_objective(self, expr):
        if expr.sense == dex.Sense.MINIMIZE:
            return self.get(expr.children[0])
        else:
            # max f(x) == min -f(x)
            mono = self.get(expr.children[0])
            return mono.negate()

    def visit_product(self, expr):
        return product_monotonicity(self, expr)

    def visit_division(self, expr):
        return division_monotonicity(self, expr)

    def visit_linear(self, expr):
        return linear_monotonicity(self, expr)

    def visit_pow(self, expr):
        return pow_monotonicity(self, expr)

    def visit_sum(self, expr):
        return linear_monotonicity(self, expr)

    def visit_abs(self, expr):
        arg = expr.children[0]
        mono = self.get(arg)

        # good examples to understand the behaviour of abs are abs(-x) and
        # abs(1/x)
        if self.is_nonnegative(arg):
            # abs(x), x > 0 is the same as x
            return mono
        elif self.is_nonpositive(arg):
            # abs(x), x < 0 is the opposite of x
            return mono.negate()
        else:
            return Monotonicity.Unknown

    def visit_nondecreasing_function(self, expr):
        return self.get(expr.children[0])

    def visit_nonincreasing_function(self, expr):
        return self.get(expr.children[0]).negate()

    def visit_sin(self, expr):
        arg = self.children[0]
        cos_bound = arg.cos()
        mono = self.get(arg)

        if mono.is_nondecreasing() and cos_bound.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bound.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nonincreasing() and cos_bound.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nondecreasing() and cos_bound.is_nonpositive():
            return Monotonicity.Nonincreasing
        else:
            return Monotonicity.Unknown

    def visit_cos(self, expr):
        arg = self.children[0]
        sin_bound = arg.sin()
        mono = self.get(arg)
        if mono.is_nonincreasing() and sin_bound.is_nonnegative():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bound.is_nonpositive():
            return Monotonicity.Nondecreasing
        elif mono.is_nondecreasing() and sin_bound.is_nonnegative():
            return Monotonicity.Nonincreasing
        elif mono.is_nonincreasing() and sin_bound.is_nonpositive():
            return Monotonicity.Nonincreasing
        else:
            return Monotonicity.Unknown

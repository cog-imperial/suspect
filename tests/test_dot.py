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

import pytest
import io
import pyomo.environ as aml
from convexity_detection.dot import *
from convexity_detection.pyomo_compat import *
import numpy as np
import pyomo.core.base.expr_pyomo4 as omo


set_pyomo4_expression_tree()


# TODO: test this
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

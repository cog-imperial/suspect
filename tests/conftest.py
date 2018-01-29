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

import hypothesis.strategies as st
from suspect import set_pyomo4_expression_tree
from suspect.bound import ArbitraryPrecisionBound as Bound
from suspect.monotonicity.monotonicity import Monotonicity
from suspect.convexity.convexity import Convexity


@st.composite
def coefficients(draw, min_value=None, max_value=None):
    return draw(st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=False,
    ))


class PlaceholderExpression(object):
    depth = 0


def bound_description_to_bound(bound_str):
    if isinstance(bound_str, str):
        return {
            'zero': Bound.zero(),
            'nonpositive': Bound(None, 0),
            'nonnegative': Bound(0, None),
            'positive': Bound(1, None),
            'negative': Bound(None, -1),
            'unbounded': Bound(None, None),
        }[bound_str]
    elif isinstance(bound_str, Bound):
        return bound_str
    else:
        return Bound(bound_str, bound_str)


def mono_description_to_mono(mono_str):
    return {
        'nondecreasing': Monotonicity.Nondecreasing,
        'nonincreasing': Monotonicity.Nonincreasing,
        'constant': Monotonicity.Constant,
        'unknown': Monotonicity.Unknown,
    }[mono_str]


def cvx_description_to_cvx(cvx_str):
    return {
        'convex': Convexity.Convex,
        'concave': Convexity.Concave,
        'linear': Convexity.Linear,
        'unknown': Convexity.Unknown,
    }[cvx_str]


def pytest_sessionstart(session):
    set_pyomo4_expression_tree()

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
from suspect.math import make_number
from suspect.float_hash import BTreeFloatHasher, RoundFloatHasher


@pytest.mark.skip('Only works with non-default Arb precision math')
def test_btree_float_hasher():
    hasher = BTreeFloatHasher()

    # fill hasher with some numbers
    for i in range(100):
        hasher.hash(make_number(i) / 10.0)
        hasher.hash(make_number(-i) / 5.0)

    h1 = hasher.hash(10.123)
    h2 = hasher.hash(10.123)
    h3 = hasher.hash(10.1234)
    h4 = hasher.hash(10.123000000001)

    assert h1 == h2
    assert h2 != h3
    assert h2 != h4


def test_round_float_hasher():
    hasher = RoundFloatHasher(3)
    assert hasher.hash(100.123) == hasher.hash(100.123)
    assert hasher.hash(100.123) == hasher.hash(100.123456)
    assert hasher.hash(10) == hasher.hash(10.0001)

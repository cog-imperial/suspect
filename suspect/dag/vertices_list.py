# Copyright 2018 Francesco Ceccon
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

"""Module containing VerticesList."""
import bisect


# pylint: disable=invalid-name
def _reverse_bisect_right(arr, x):
    """Like bisect.bisect_right, but insert in a descending list"""
    lo = 0
    hi = len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if x > arr[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


class VerticesList(object):
    """A list of vertices sorted by their depth."""
    def __init__(self, reverse=False):
        self._vertices = []
        self._vertices_depth = []
        self._vertices_set = set()
        self._reverse = reverse

        if self._reverse:
            self._find_insertion_idx = _reverse_bisect_right
        else:
            self._find_insertion_idx = bisect.bisect_right

    def append(self, vertex, depth):
        """Append vertex to the list, keeping the vertices sorted by depth"""
        if id(vertex) in self._vertices_set:
            return
        insertion_idx = self._find_insertion_idx(self._vertices_depth, depth)
        self._vertices.insert(insertion_idx, vertex)
        self._vertices_depth.insert(insertion_idx, depth)
        self._vertices_set.add(id(vertex))

    def pop(self):
        """Pop an element from the front of the list"""
        self._vertices_depth.pop(0)
        vertex = self._vertices.pop(0)
        if id(vertex) in self._vertices_set:
            self._vertices_set.remove(id(vertex))
        return vertex

    def __iter__(self):
        return iter(self._vertices)

    def __len__(self):
        return len(self._vertices)

    def __contains__(self, vertex):
        return id(vertex) in self._vertices_set

    def __bool__(self):
        return len(self) > 0

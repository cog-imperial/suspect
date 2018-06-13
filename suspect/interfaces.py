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

"""Interfaces used in SUSPECT.

You don't need them unless you are using a type checker on your codebase.
"""
import abc
from typing import Any, Generic, List, TypeVar


V = TypeVar('V')
C = TypeVar('C')


class Problem(Generic[V]):
    """Generic problem with vertices of type V."""
    pass


class Visitor(Generic[V, C], metaclass=abc.ABCMeta):
    """Visitor for vertices of type V."""
    @abc.abstractmethod
    def visit(self, vertex: V, ctx: C) -> bool:
        """Visit vertex. Return True if the vertex should be considered "dirty"."""
        pass


class Iterator(Generic[V, C], metaclass=abc.ABCMeta):
    """Iterator over vertices of Problem[V]."""
    @abc.abstractmethod
    def iterate(self, problem: Problem[V], visitor: Visitor[V, C], ctx: C,
                *args: Any, **kwargs: Any) -> List[V]:
        """Iterate over vertices of problem, calling visitor on each one of them.

        Returns the list of vertices for which the visitor returned a True value.
        """
        pass


class ForwardIterator(Iterator[V, C]):
    """An iterator for iterating over nodes in ascending depth order."""
    pass


class BackwardIterator(Iterator[V, C]):
    """An iterator for iterating over nodes in descending depth order."""
    pass

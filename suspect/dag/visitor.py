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

"""Visitor implementation for ProblemDag."""
import abc
from typing import Any, Callable, Dict, Generic, TypeVar
from suspect.util import deprecated
from suspect.interfaces import Visitor as IVisitor, C
from suspect.dag.expressions import Expression


R = TypeVar('R')


class Visitor(Generic[R, C], IVisitor[Expression, C]):
    """Visitor over vertices of ProblemDag."""
    def __init__(self, *_args: Any) -> None:
        try:
            self._registered_callbacks = self.register_callbacks()
        except NotImplementedError:
            self._registered_callbacks = self.register_handlers()

    # @abc.abstractmethod
    def register_callbacks(self) -> Dict[type, Callable[[Expression, C], R]]:
        """Register callbacks for each expression type."""        
        raise NotImplementedError()

    @deprecated('register_callbacks')
    def register_handlers(self) -> Dict[type, Callable[[Expression, C], R]]:
        """Register callbacks for each expression type.

        DEPRECATED: use register_callbacks instead.
        """
        pass
        # return self.register_callbacks()

    @abc.abstractmethod
    def handle_result(self, expr: Expression, result: R, ctx: C) -> bool:
        """Handle visit result."""
        pass

    def _handle_result(self, expr: Expression, result: R, ctx: C) -> bool:
        return self.handle_result(expr, result, ctx)

    def _visit_expr(self, expr: Expression, ctx: C, cb: Callable[[Expression, C], R]) -> bool:
        result = cb(expr, ctx)
        return self._handle_result(expr, result, ctx)

    def visit(self, expr: Expression, ctx: C) -> bool: # pylint: disable=missing-docstring
        type_ = type(expr)
        cb = self._registered_callbacks.get(type_)
        if cb is not None:
            return self._visit_expr(expr, ctx, cb)

        # try superclasses, for most cases this will work fine
        # but since dicts are not ordered it could cause
        # unexpected behaviour
        for target_type, cb in self._registered_callbacks.items():
            if isinstance(expr, target_type):
                return self._visit_expr(expr, ctx, cb)
        return True


class ForwardVisitor(Generic[R, C], Visitor[R, C]): # pylint: disable=missing-docstring,abstract-method
    """Visitor when visiting ProblemDag forward."""
    pass


class BackwardVisitor(Generic[R, C], Visitor[R, C]): # pylint: disable=missing-docstring,abstract-method
    """Visitor when visiting ProblemDag backward."""
    def _handle_result(self, _expr: Expression, result: R, ctx: C) -> bool:
        if result is None:
            return False
        any_change = False
        for child, value in result.items():
            any_change |= self.handle_result(child, value, ctx)
        return any_change

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
import abc


class Visitor(object, metaclass=abc.ABCMeta):
    def __init__(self, *args):
        self._registered_handlers = self.register_handlers()

    @abc.abstractmethod
    def register_handlers(self):
        pass

    @abc.abstractmethod
    def handle_result(self, expr, result, ctx):
        pass

    def visit(self, expr, ctx):
        type_ = type(expr)
        cb = self._registered_handlers.get(type_)
        if cb is not None:
            return cb(expr, ctx)

        # try superclasses, for most cases this will work fine
        # but since dicts are not ordered it could cause
        # unexpected behaviour
        for target_type, cb in self._registered_handlers.items():
            if isinstance(expr, target_type):
                return cb(expr, ctx)

    def __call__(self, expr, ctx):
        return self.visit(expr, ctx)


class ForwardVisitor(Visitor):
    def __call__(self, expr, ctx):
        result = self.visit(expr, ctx)
        if result is not None:
            has_changed = self.handle_result(expr, result, ctx)
            if has_changed:
                return True
        return False


class BackwardVisitor(Visitor):
    @abc.abstractmethod
    def handle_result(self, expr, result, ctx):
        pass

    def __call__(self, expr, ctx):
        result = self.visit(expr, ctx)
        if isinstance(result, dict):
            any_changed = False
            for k, v in result.items():
                has_changed = self.handle_result(k, v, ctx)
                if has_changed:
                    any_changed = True
            return any_changed
        elif result is not None:
            raise RuntimeError('BackwardVisitor must return dict')
        return False

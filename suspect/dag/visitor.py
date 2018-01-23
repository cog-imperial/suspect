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


class Dispatcher(object):
    def __init__(self, lookup, allow_missing=False):
        self._lookup = lookup
        self._allow_missing = allow_missing

    def dispatch(self, expr):
        type_ = type(expr)
        cb = self._lookup.get(type_)
        if cb is not None:
            return cb(expr)

        # try superclasses, for most cases this will work fine
        # but since dicts are not ordered it could cause
        # unexpected behaviour
        for target_type, cb in self._lookup.items():
            if isinstance(expr, target_type):
                return cb(expr)

        if not self._allow_missing:
            raise RuntimeError('Could not find callback for {} of type {}'.format(expr, type_))

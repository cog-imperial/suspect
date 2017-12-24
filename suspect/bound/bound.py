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
import warnings


class Bound(metaclass=abc.ABCMeta):
    @abc.abstractproperty
    def lower_bound(self):
        pass

    @property
    def l(self):
        warnings.warn(
            "Bound.l is deprecated. Use Bound.lower_bound",
            DeprecationWarning,
        )
        return self.lower_bound

    @abc.abstractproperty
    def upper_bound(self):
        pass

    @property
    def u(self):
        warnings.warn(
            "Bound.u is deprecated. Use Bound.upper_bound",
            DeprecationWarning,
        )
        return self.upper_bound

    @abc.abstractmethod
    def is_zero(self):
        pass

    @abc.abstractmethod
    def is_positive(self):
        pass

    @abc.abstractmethod
    def is_negative(self):
        pass

    @abc.abstractmethod
    def tighten(self, other):
        """Returns a new bound which is the tightest intersection of bounds."""
        pass

    @abc.abstractmethod
    def add(self, other):
        pass

    @abc.abstractmethod
    def sub(self, other):
        pass

    @abc.abstractmethod
    def mul(self, other):
        pass

    @abc.abstractmethod
    def div(self, other):
        pass

    @abc.abstractmethod
    def equals(self, other):
        pass

    @abc.abstractmethod
    def contains(self, other):
        pass

    @abc.abstractmethod
    def zero(self):
        pass

    @abc.abstractmethod
    def is_nonpositive(self):
        pass

    @abc.abstractmethod
    def is_nonnegative(self):
        pass

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.sub(other)

    def __neg__(self):
        return self.zero() - self

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self.div(other)

    def __eq__(self, other):
        return self.equals(other)

    def __contains__(self, other):
        return self.contains(other)

    def __repr__(self):
        return '<{} at {}>'.format(str(self), id(self))

    def __str__(self):
        return '[{}, {}]'.format(self.lower_bound, self.upper_bound)

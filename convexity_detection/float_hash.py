import abc
from convexity_detection.math import (
    mpf,
    almosteq,
    almostgte,
    almostlte,
)


class FloatHasher(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def hash(self, f):
        raise NotImplementedError('hash')


class BTreeFloatHasher(FloatHasher):
    """A floating point hasher that keeps all seen floating
    point numbers ina binary tree.

    Good if the unique values of the floating point numbers in
    the problem are relatively few.
    """

    class Node(object):
        def __init__(self, num, hash_, left=None, right=None):
            self.num = num
            self.hash = hash_
            self.left = left
            self.right = right

    def __init__(self):
        self.root = None
        self.node_count = 0

    def hash(self, f):
        f = mpf(f)
        if self.root is None:
            self.root = self._make_node(f)
            return self.root.hash

        curr_node = self.root
        while True:
            if almosteq(f, curr_node.num):
                return curr_node.hash
            elif almostlte(f, curr_node.num):
                if curr_node.left is None:
                    new_node = self._make_node(f)
                    curr_node.left = new_node
                    return new_node.hash
                else:
                    curr_node = curr_node.left
            else:
                if curr_node.right is None:
                    new_node = self._make_node(f)
                    curr_node.right = new_node
                    return new_node.hash
                else:
                    curr_node = curr_node.right

    def _make_node(self, f):
        node = self.Node(f, self.node_count, None, None)
        self.node_count += 1
        return node


class RoundFloatHasher(FloatHasher):
    """A float hasher that hashes floats up to the n-th
    decimal place.
    """
    def __init__(self, n=2):
        self.n = 10**n

    def hash(self, f):
        return hash(int(f * self.n))

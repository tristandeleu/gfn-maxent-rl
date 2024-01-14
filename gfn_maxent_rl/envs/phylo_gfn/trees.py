import numpy as np

from dataclasses import dataclass
from typing import Union
from functools import cached_property


@dataclass(frozen=True)
class Leaf:
    index: int
    sequence: np.ndarray

    def __hash__(self):
        return hash((self.index, self.sequence))

    def __eq__(self, other):
        return ((self.index == other.index)
            and np.all(self.sequence == other.sequence))

    def __str__(self):
        return str(self.index)

    def to_tuple(self):
        return (self.index,)


@dataclass(frozen=True)
class RootedTree:
    left: Union['Leaf', 'RootedTree']
    right: Union['Leaf', 'RootedTree']

    @cached_property
    def mutations(self):
        overlap = self.left.sequence & self.right.sequence
        return np.sum(overlap == 0)

    @cached_property
    def sequence(self):
        overlap = self.left.sequence & self.right.sequence
        union = self.left.sequence | self.right.sequence
        return np.where(overlap > 0, overlap, union)

    def __hash__(self):
        return hash(frozenset({self.left, self.right}))

    def __eq__(self, other):
        return ({self.left, self.right} == {other.left, other.right})

    def __str__(self):
        left_str, right_str = str(self.left), str(self.right)
        left_str, right_str = left_str.split('\n'), right_str.split('\n')

        data = []
        data.append('┬─' + (' ' * (len(left_str) == 1)) + left_str[0])
        data.extend(['│ ' + p for p in left_str[1:]])

        data.append('└─' + (' ' * (len(right_str) == 1)) + right_str[0])
        data.extend(['  ' + p for p in right_str[1:]])

        return '\n'.join(data)

    def to_tuple(self):
        # Returns the structure of the tree in a pre-order representation
        # Encoding: "internal node" = -1
        return (-1,) + self.left.to_tuple() + self.right.to_tuple()

    @classmethod
    def from_tuple(cls, tup, sequences):
        def _from_tuple(index):
            node = tup[index]
            if node >= 0:
                return (Leaf(index=node, sequence=sequences[node]), index)
            else:
                left, index = _from_tuple(index + 1)
                right, index = _from_tuple(index + 1)
                return (cls(left=left, right=right), index)
        tree, _ = _from_tuple(0)
        return tree

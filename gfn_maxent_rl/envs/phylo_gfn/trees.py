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

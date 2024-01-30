import numpy as np

from dataclasses import dataclass
from typing import Union
from functools import cached_property
from numpy.random import default_rng


@dataclass(frozen=True)
class Leaf:
    index: int
    sequence: np.ndarray

    def __hash__(self):
        return hash((self.index, tuple(self.sequence)))

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

    @cached_property
    def index(self):
        return min(self.left.index, self.right.index)

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
        if all(x == -2 for x in tup):
            return None  # Intermediate state

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


def generate_trajectories(tree, num_nodes, num_trajectories, rng=default_rng()):
    if isinstance(tree, Leaf):
        trajectories = np.zeros((num_trajectories, 0), dtype=np.int_)
    else:
        # Get the last action
        action = tree.left.index * num_nodes \
            - tree.left.index * (tree.left.index + 1) // 2 \
            + tree.right.index - (tree.left.index + 1)
        actions = np.full((num_trajectories, 1), action, dtype=np.int_)

        # Get trajectories & number of trajectories
        left_trajs = generate_trajectories(tree.left, num_nodes, num_trajectories, rng=rng)
        right_trajs = generate_trajectories(tree.right, num_nodes, num_trajectories, rng=rng)

        # Shuffle the orders
        left_num, right_num = left_trajs.shape[1], right_trajs.shape[1]
        masks = np.zeros((num_trajectories, left_num + right_num), dtype=np.bool_)
        masks[:, :left_num] = True
        masks = rng.permuted(masks, axis=1)

        # Interlace actions
        prefix = np.zeros((num_trajectories, left_num + right_num), dtype=np.int_)
        prefix[masks] = left_trajs.reshape(-1)
        prefix[~masks] = right_trajs.reshape(-1)

        trajectories = np.concatenate((prefix, actions), axis=1)

    return trajectories


def get_log_backward_prob(actions):
    batch_size, num_nodes = actions.shape
    lefts, rights = np.triu_indices(num_nodes + 1, k=1)
    arange = np.arange(batch_size)

    types = np.ones((batch_size, num_nodes + 1), dtype=np.int_)
    log_pB = np.zeros((batch_size,), dtype=np.float_)
    for action in actions.T:
        left, right = lefts[action], rights[action]

        # Update the types
        types[arange, left] = 2  # Rooted tree
        types[arange, right] = 0  # Padding

        # Update the log-probabilities
        log_pB -= np.log(np.sum(types == 2, axis=1))

    return log_pB


if __name__ == '__main__':
    tree = RootedTree(
        left=RootedTree(
            left=Leaf(0, None),
            right=Leaf(1, None)
        ),
        right=RootedTree(
            left=Leaf(2, None),
            right=RootedTree(
                left=Leaf(3, None),
                right=Leaf(4, None)
            )
        )
    )
    print(tree)

    rng = default_rng(0)
    actions = generate_trajectories(tree, 5, 5, rng)
    log_pB = get_log_backward_prob(actions)

    print(actions)
    print(log_pB)

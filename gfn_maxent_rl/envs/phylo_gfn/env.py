import numpy as np
import gym

from gym.spaces import Space, Dict, Box, Discrete

from gfn_maxent_rl.envs.phylo_gfn.trees import Leaf, RootedTree


CHARACTERS_MAPS = {
    'DNA': {'A': 0b1000, 'C': 0b0100, 'G': 0b0010, 'T': 0b0001, 'N': 0b1111},
    'RNA': {'A': 0b1000, 'C': 0b0100, 'G': 0b0010, 'U': 0b0001, 'N': 0b1111},
    'DNA_WITH_GAP': {
        'A': 0b10000, 'C': 0b01000, 'G': 0b00100, 'T': 0b00010,
        '-': 0b00001, 'N': 0b11110
    },
    'RNA_WITH_GAP': {
        'A': 0b10000, 'C': 0b01000, 'G': 0b00100, 'U': 0b00010,
        '-': 0b00001, 'N': 0b11110
    }
}

class RootedTreeSpace(Space[RootedTree]):
    def contains(self, x):
        return isinstance(x, (RootedTree, Leaf))


class PhyloTreeEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, sequences, sequence_type='DNA', C=0., scale=1.):
        char_dict = CHARACTERS_MAPS[sequence_type]
        self.sequences = np.array([[char_dict[c] for c in sequence]
            for sequence in sequences.values()], dtype=np.int_)
        num_nodes, sequence_length = self.sequences.shape
        assert num_nodes > 1

        self.sequence_type = sequence_type
        self.C = C
        self.scale = scale
        self._state = None
        self._lefts, self._rights = np.triu_indices(num_nodes, k=1)
        self._reward_offset = (C / scale) / num_nodes

        max_actions = num_nodes * (num_nodes - 1) // 2
        observation_space = Dict({
            'sequences': Box(low=0., high=1.,
                shape=(num_nodes, sequence_length, 5), dtype=np.float32),
            'length': Box(low=1, high=num_nodes, shape=(), dtype=np.int_),
            'tree': RootedTreeSpace(),
            'mask': Box(low=0., high=1., shape=(max_actions,), dtype=np.float32),
        })
        action_space = Discrete(max_actions + 1)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, *, seed=None, options=None):
        max_actions = self.single_action_space.n - 1
        # Initialize the state with only leaves and all the actions being valid
        self._state = {
            'trees': [[Leaf(index=n, sequence=seq)
                for (n, seq) in enumerate(self.sequences)]
                for _ in range(self.num_envs)],
            'masks': np.ones((self.num_envs, max_actions), dtype=np.bool_)
        }
        return (self.observations(), {})

    def step(self, actions):
        stop_action = self.single_action_space.n - 1  # num_nodes * (num_nodes - 1) // 2
        dones = (actions == stop_action)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)

        rewards = np.zeros((self.num_envs,), dtype=np.float_)
        for i, (trees, action) in enumerate(zip(self._state['trees'], actions)):
            if action == stop_action:                
                if any((tree is not None) for tree in trees[1:]):
                    raise RuntimeError('Invalid action: called the stop '
                        'action even though we are not in a terminating state.')

                # Reset the state (zero reward at the final step)
                trees = [Leaf(index=n, sequence=seq)
                    for (n, seq) in enumerate(self.sequences)]
                self._state['masks'][i, :] = True

            else:
                # Get the indices of the two trees to be merged
                left, right = self._lefts[action], self._rights[action]

                # Merge the two trees into the "left" one
                trees[left] = RootedTree(left=trees[left], right=trees[right])
                trees[right] = None  # Remove the "right" tree

                # Intermediate log-reward: the sum of intermediate log-reward
                # is: (C - total_mutations) / scale
                rewards[i] = self._reward_offset - trees[left].mutations / self.scale

                # The "right" tree is no longer valid for merging
                self._state['masks'][i, self._lefts == right] = False
                self._state['masks'][i, self._rights == right] = False

        return (self.observations(), rewards, dones, truncated, {})

    def observations(self):
        # Construct the sequence features
        shape = self.sequences.shape
        sequences = np.zeros((self.num_envs,) + shape + (5,), dtype=np.float32)
        for i in range(self.num_envs):
            for j, tree in enumerate(self._state['trees'][i]):
                if tree is None:
                    continue
                sequences[i, j] = ((tree.sequence[:, None] & (1 << np.arange(5))) > 0)

        return {
            # The Fitch features of each tree
            'sequences': sequences,  # (num_envs, num_nodes, sequence_length, 5)

            # The number of valid trees in the "sequences" (among the "num_nodes")
            'length': np.array([len([tree is not None for tree in trees])
                for trees in self._state['trees']], dtype=np.int_),  # (num_envs,)

            # Only returning "trees[0]" because at the terminating state,
            # the final tree will be in trees[0], and this key will only be
            # useful when sampling at test time.
            'tree': tuple(trees[0] for trees in self._state['trees']),

            # The mask for the valid actions: (num_envs, num_nodes * (num_nodes - 1) // 2)
            'mask': np.copy(self._state['masks']).astype(np.float32)
        }


if __name__ == '__main__':
    from numpy.random import default_rng
    import json

    def random_actions(observations, rng=default_rng()):
        # Get the action mask from the mask returned by the observations
        action_masks = observations['mask'].astype(np.bool_)
        is_terminal = np.any(action_masks, axis=1, keepdims=True)
        action_masks = np.concatenate((action_masks, ~is_terminal), axis=1)

        # Compute uniform (log-)probabilities
        logits = np.where(action_masks, 1., -np.inf)

        # Apply the Gumbel-max trick to sample from categorical distribution
        # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
        z = rng.gumbel(loc=0., scale=1., size=logits.shape)
        actions = np.argmax(logits + z, axis=1)

        return actions

    rng = default_rng(1)
    with open('gfn_maxent_rl/envs/phylo_gfn/datasets/DS1.json', 'r') as f:
        sequences = json.load(f)
    env = PhyloTreeEnvironment(num_envs=1, sequences=sequences, sequence_type='DNA_WITH_GAP')
    dones = np.zeros((env.num_envs,), dtype=np.bool_)

    observations, _ = env.reset()
    while not np.all(dones):
        actions = random_actions(observations, rng=rng)
        observations, rewards, dones, _, _ = env.step(actions)

    print(observations['tree'][0])

import numpy as np
import gym
import math
import operator as op

from gym.spaces import Dict, Box, Discrete
from functools import reduce

from gfn_maxent_rl.envs.phylo_gfn.trees import Leaf, RootedTree
from gfn_maxent_rl.envs.phylo_gfn.utils import CHARACTERS_MAPS, get_tree_type
from gfn_maxent_rl.envs.phylo_gfn.policy import uniform_log_policy, action_mask
from gfn_maxent_rl.envs.errors import StatesEnumerationError


class PhyloTreeEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, sequences, reward, sequence_type='DNA'):
        char_dict = CHARACTERS_MAPS[sequence_type]
        self.sequences = np.array([[char_dict[c] for c in sequence]
            for sequence in sequences.values()], dtype=np.int_)
        num_nodes, sequence_length = self.sequences.shape
        assert num_nodes > 1

        self.reward = reward
        self.sequence_type = sequence_type
        self._state = None
        self._lefts, self._rights = np.triu_indices(num_nodes, k=1)

        max_actions = num_nodes * (num_nodes - 1) // 2
        observation_space = Dict({
            'sequences': Box(low=0., high=1.,
                shape=(num_nodes, sequence_length, 5), dtype=np.float32),
            'type': Box(low=0, high=2, shape=(num_nodes,), dtype=np.int_),
            'tree': Box(low=-2, high=num_nodes, shape=(2 * num_nodes - 1,), dtype=np.int_),
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
                self._state['trees'][i] = [Leaf(index=n, sequence=seq)
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
                rewards[i] = self.reward.delta_score(trees[left])

                # The "right" tree is no longer valid for merging
                self._state['masks'][i, self._lefts == right] = False
                self._state['masks'][i, self._rights == right] = False

        return (self.observations(), rewards, dones, truncated, {})

    def observations(self):
        # Construct the sequence features
        sequences = np.zeros(self.observation_space['sequences'].shape, dtype=np.float32)
        trees = np.full(self.observation_space['tree'].shape, -2, dtype=np.int_)

        for i in range(self.num_envs):
            state = self._state['trees'][i]
            for j, tree in enumerate(state):
                if tree is None:
                    continue
                sequences[i, j] = ((tree.sequence[:, None] & (1 << np.arange(5))) > 0)

            if all(tree is None for tree in state[1:]):
                trees[i] = state[0].to_tuple()

        return {
            # The Fitch features of each tree
            'sequences': sequences,  # (num_envs, num_nodes, sequence_length, 5)

            # The number of valid trees in the "sequences" (among the "num_nodes")
            'type': np.array([[get_tree_type(tree) for tree in trees]
                for trees in self._state['trees']], dtype=np.int_),  # (num_envs, num_nodes)

            # A representation of the structure of the trees ("-2" = intermediate state)
            'tree': trees,  # (num_envs, 2 * num_nodes - 1)

            # The mask for the valid actions: (num_envs, num_nodes * (num_nodes - 1) // 2)
            'mask': np.copy(self._state['masks']).astype(np.float32)
        }
    
    # Properties & methods to interact with the replay buffer

    @property
    def observation_dtype(self):
        num_nodes, sequence_length = self.sequences.shape
        nbytes_seq = math.ceil((num_nodes * sequence_length * 5) / 8)
        nbytes_mask = math.ceil((self.single_action_space.n - 1) / 8)  # num_nodes * (num_nodes - 1) // 2
        return np.dtype([
            ('sequences', np.uint8, (nbytes_seq,)),
            ('type', np.int_, (num_nodes,)),
            ('mask', np.uint8, (nbytes_mask,)),
        ])

    @property
    def max_length(self):
        return self.sequences.shape[0] + 1

    def encode(self, observations):
        batch_size = observations['sequences'].shape[0]

        def _encode(decoded):
            encoded = decoded.reshape(batch_size, -1)
            return np.packbits(encoded.astype(np.int32), axis=1)

        encoded = np.empty((batch_size,), dtype=self.observation_dtype)
        encoded['sequences'] = _encode(observations['sequences'])
        encoded['type'] = observations['type']
        encoded['mask'] = _encode(observations['mask'])
        return encoded

    def _decode(self, encoded, shape):
        count = reduce(op.mul, shape, 1)
        decoded = np.unpackbits(encoded, axis=-1, count=count)
        decoded = decoded.reshape(*encoded.shape[:-1], *shape)
        return decoded.astype(np.float32)

    def decode(self, observations):
        return {
            'sequences': self._decode(observations['sequences'],
                self.single_observation_space['sequences'].shape),
            'type': observations['type'],
            'mask': self._decode(observations['mask'],
                self.single_observation_space['mask'].shape)
        }

    def decode_sequence(self, samples):
        return self.decode(samples['observations'])

    # Method to interact with the algorithm (uniform sampling of action)

    def uniform_log_policy(self, observations):
        return uniform_log_policy(observations['mask'])

    def num_parents(self, observations):
        # Elements with type "2" correspond to rooted trees. The number
        # of parents is the number of rooted trees (not leaves).
        return (observations['type'] == 2).sum(axis=-1)

    def action_mask(self, observations):
        return action_mask(observations['mask'])

    # Method for evaluation

    def all_states_batch_iterator(self, batch_size, terminating=False):
        raise StatesEnumerationError('Impossible to enumerate all the '
            'states of `PhyloTreeEnvironment`.')

    def log_reward(self, observations):
        return self.reward.log_reward(observations['tree'])

    @property
    def mdp_state_graph(self):
        raise StatesEnumerationError('Impossible to enumerate all the '
            'states of `PhyloTreeEnvironment`.')

    def observation_to_key(self, observations):
        return [RootedTree.from_tuple(tree, self.sequences)
            for tree in observations['tree']]


if __name__ == '__main__':
    from numpy.random import default_rng
    import json
    from gfn_maxent_rl.envs.phylo_gfn.rewards import ExponentialReward

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
    reward = ExponentialReward(num_nodes=len(sequences))
    env = PhyloTreeEnvironment(
        num_envs=1,
        sequences=sequences,
        reward=reward,
        sequence_type='DNA_WITH_GAP'
    )
    dones = np.zeros((env.num_envs,), dtype=np.bool_)

    observations, _ = env.reset()
    # while not np.all(dones):
    for _ in range(100):
        actions = random_actions(observations, rng=rng)
        observations, rewards, dones, _, _ = env.step(actions)

    print(RootedTree.from_tuple(observations['tree'][0], env.sequences))

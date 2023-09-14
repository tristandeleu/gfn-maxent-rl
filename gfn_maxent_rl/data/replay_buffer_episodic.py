import numpy as np
import math

from numpy.random import default_rng

from gfn_maxent_rl.data.replay_buffer import ReplayBuffer
from gfn_maxent_rl.envs.dag_gfn.jraph_utils import batch_sequences_to_graphs_tuple


class EpisodicReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        self.max_length = (num_variables * (num_variables - 1) // 2) + 1
        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            # State
            ('adjacencies', np.uint8, (self.max_length, nbytes)),
            ('masks', np.uint8, (self.max_length, nbytes)),
            # Action
            ('actions', np.int_, (self.max_length,)),
            # Reward
            ('rewards', np.float_, (self.max_length,)),
            # Misc
            ('is_complete', np.bool_, ()),
            ('length', np.int_, ()),
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0

    def add(self, observations, actions, rewards, dones, next_observations, indices=None):
        if indices is None:
            indices = self._index + np.arange(actions.shape[0])
            self._index += actions.shape[0]

        # Add data to the episode buffer
        episode_idx = self._replay['length'][indices]
        self._replay['adjacencies'][indices, episode_idx] = self.encode(observations['adjacency'])
        self._replay['masks'][indices, episode_idx] = self.encode(observations['mask'])
        self._replay['actions'][indices, episode_idx] = actions
        self._replay['rewards'][indices, episode_idx] = rewards

        # Set complete episodes & update episode indices
        self._replay['is_complete'][indices[dones]] = True
        self._replay['length'][indices[~dones]] += 1

        # Get new indices for new trajectories, and clear data already present
        num_dones = np.sum(dones)
        new_indices = (self._index + np.arange(num_dones)) % self.capacity
        self._replay[new_indices] = 0  # Clear data

        # Set the new indices for the next trajectories
        indices[dones] = new_indices
        self._index = (self._index + num_dones) % self.capacity

        return indices

    def sample(self, batch_size, rng=default_rng()):
        indices = rng.choice(self.capacity, batch_size,
            replace=False, p=self._replay['is_complete'] / len(self))
        samples = self._replay[indices]

        actions, lengths = samples['actions'], samples['length']

        return {
            'observations': {
                'adjacency': self.decode(samples['adjacencies']),
                'graph': batch_sequences_to_graphs_tuple(
                    self.num_variables, actions, lengths),
                'mask': self.decode(samples['masks']),
            },
            'actions': actions,
            'rewards': samples['rewards'],
            'lengths': lengths,
        }

    def __len__(self):
        return np.sum(self._replay['is_complete'])

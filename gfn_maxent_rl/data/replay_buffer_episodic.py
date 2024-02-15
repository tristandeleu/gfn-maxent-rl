import numpy as np

from numpy.random import default_rng

from gfn_maxent_rl.data.replay_buffer import ReplayBuffer


class EpisodicReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.env = env

        dtype = np.dtype([
            ('observations', env.observation_sequence_dtype, (env.max_length,)),
            ('actions', np.int_, (env.max_length,)),
            ('rewards', np.float_, (env.max_length,)),

            # Misc
            ('is_complete', np.bool_, ()),
            ('lengths', np.int_, ()),
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0

    def add(self, observations, actions, rewards, dones, next_observations, indices=None):
        if indices is None:
            indices = self._index + np.arange(actions.shape[0])
            self._index += actions.shape[0]

        # Add data to the episode buffer
        episode_idx = self._replay['lengths'][indices]
        self._replay['observations'][indices, episode_idx] = self.env.encode_sequence(observations)
        self._replay['actions'][indices, episode_idx] = actions
        self._replay['rewards'][indices, episode_idx] = rewards

        # Set complete episodes & update episode indices
        self._replay['is_complete'][indices[dones]] = True
        self._replay['lengths'][indices[~dones]] += 1

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

        return {
            'observations': self.env.decode_sequence(samples),
            'actions': samples['actions'],
            'rewards': samples['rewards'],
            'lengths': samples['lengths'],
        }

    def __len__(self):
        return np.sum(self._replay['is_complete'])

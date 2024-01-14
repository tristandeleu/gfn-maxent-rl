import numpy as np

from numpy.random import default_rng


class ReplayBuffer:
    def __init__(self, capacity, env):
        self.capacity = capacity
        self.env = env

        dtype = np.dtype([
            ('observation', env.observation_dtype),
            ('action', np.int_, (1,)),
            ('reward', np.float_, (1,)),
            ('next_observation', env.observation_dtype)
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False

    def add(self, observations, actions, rewards, dones, next_observations, indices=None):
        if np.all(dones):
            return None

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity

        self._replay['observation'][add_idx] = self.env.encode(observations)[~dones]
        self._replay['action'][add_idx] = np.asarray(actions[~dones].reshape(-1, 1), dtype=np.int_)
        self._replay['reward'][add_idx] = np.asarray(rewards[~dones].reshape(-1, 1), dtype=np.float_)
        self._replay['next_observation'][add_idx] = self.env.encode(next_observations)[~dones]

        return None

    def sample(self, batch_size, rng=default_rng()):
        indices = rng.choice(len(self), size=batch_size, replace=False)
        samples = self._replay[indices]

        return {
            'observation': self.env.decode(samples['observation']),
            'action': samples['action'],
            'reward': samples['reward'],
            'next_observation': self.env.decode(samples['next_observation']),
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    def can_sample(self, batch_size):
        return len(self) >= batch_size

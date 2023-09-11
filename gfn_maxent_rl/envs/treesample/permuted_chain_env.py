import numpy as np
import gym

from gym.spaces import Box, Discrete
from numpy.random import default_rng


class PermutedChainEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, cond_distributions, permutation):
        self.cond_distributions = cond_distributions
        self.permutation = permutation
        self.num_variables, self.num_categories = cond_distributions.shape[:2]

        self._state = np.full((num_envs, self.num_variables), -1, dtype=np.int_)
        self._index = np.zeros((num_envs,), dtype=np.int_)
        self._arange = np.arange(num_envs)

        observations_space = Box(
            low=-1,
            high=self.num_categories,
            shape=(self.num_variables,),
            dtype=np.int_
        )
        action_space = Discrete(self.num_categories)
        super().__init__(num_envs, observations_space, action_space)

    def reset(self, *, seed=None, options=None):
        self._state[:] = -1
        self._index[:] = 0
        return (np.copy(self._state), {})

    def step(self, actions):
        indices = self.permutation[self._index]
        self._state[self._arange, indices] = actions

        # Compute the rewards (more precisely, difference in log-rewards)
        rewards = np.zeros((self.num_envs,), dtype=np.float_)

        # If the binary potential to the left of the variable can be added
        left = np.logical_and(
            indices > 0,
            self._state[self._arange, indices - 1] != -1
        )
        index_left = indices[left] - 1
        values_left = self._state[left, index_left]
        rewards[left] += np.log(self.cond_distributions[index_left, values_left, actions[left]])

        # If the binary potential to the right of the variable can be added
        right = np.logical_and(
            indices < self.num_variables - 1,
            self._state[self._arange, (indices + 1) % self.num_variables] != -1
        )
        index_right = indices[right]
        values_right = self._state[right, index_right + 1]
        rewards[right] += np.log(self.cond_distributions[index_right, actions[right], values_right])

        self._index = (self._index + 1) % self.num_variables
        dones = (self._index == 0)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)

        return (np.copy(self._state), rewards, dones, truncated, {})

    @classmethod
    def random(
        cls,
        num_envs,
        num_variables,
        num_categories,
        rng=default_rng(),
        alpha=1.,
    ):
        # Generate binary factors from symmetric Dirichlet
        alphas = np.full((num_categories,), alpha)
        cond_distributions = rng.dirichlet(alphas, size=(num_variables, num_categories))

        # Sample a random permutation of the variables
        permutation = rng.permutation(num_variables)

        return cls(num_envs, cond_distributions, permutation)

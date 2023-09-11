import numpy as np
import gym
import math

from gym.spaces import Box, Discrete
from numpy.random import default_rng


class ChainEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, unary_potentials, factor=2.5):
        self.unary_potentials = unary_potentials
        self.factor = factor
        self.num_variables, self.num_categories = unary_potentials.shape

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
        self._state[self._arange, self._index] = actions

        # Compute the rewards (more precisely, difference in log-rewards)
        rewards = self.unary_potentials[self._index, actions]  # Unary potentials

        # Binary potentials
        mid = (self._index > 0)
        value_tm1 = self._state[mid, self._index[mid] - 1]
        value_t = self._state[mid, self._index[mid]]
        diffs = np.abs(value_t - value_tm1)
        diffs = np.where(diffs == self.num_categories - 1, 1, diffs)  # Difference on 1D torus
        rewards[mid] += self.factor * diffs

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
        rbf_bandwidth=1.,
        rbf_scale=0.5,
        **kwargs
    ):
        # Create the covariance matrix
        x, y = np.arange(num_variables), np.arange(num_categories)
        X, Y = np.meshgrid(x, y)
        XY = np.dstack([X, Y]).reshape(-1, 2)
        dists = np.sum((XY[:, None] - XY) ** 2, axis=2)
        covariance = rbf_scale * np.exp(-0.5 * dists / rbf_bandwidth)

        # Sample unary potentials from a GP prior
        unary_potentials = rng.multivariate_normal(
            mean=np.zeros((covariance.shape[0],)),
            cov=covariance
        )
        unary_potentials = unary_potentials.reshape(num_variables, num_categories)
        return cls(num_envs, unary_potentials, **kwargs)

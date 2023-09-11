import numpy as np
import gym
import math

from gym.spaces import Dict, Box, MultiDiscrete, MultiBinary


class FactorGraphEnvironment(gym.vector.VectorEnv):
    def __init__(self, num_envs, graph, potentials, permutation):
        self.graph = graph
        self.potentials = potentials
        self.permutation = permutation
        self.num_variables = len(permutation)
        
        # Get the number of categories
        clique, potential = potentials[0]
        self.num_categories = round(math.exp(math.log(len(potential)) / len(clique)))
        for clique, potential in potentials:
            assert potential.size == self.num_categories ** len(clique)

        self._state = np.full((num_envs, self.num_variables), -1, dtype=np.int_)
        self._arange = np.arange(num_envs)

        observation_space = Dict({
            'variables': Box(
                low=-1,
                high=self.num_categories,
                shape=(self.num_variables,),
                dtype=np.int_
            ),
            'mask': MultiBinary(self.num_variables),
        })
        action_space = MultiDiscrete([self.num_variables, self.num_categories])
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, *, seed=None, options=None):
        self._state[:] = -1
        return (self.observations(), {})

    def step(self, actions):
        # Clear complete episodes
        is_complete = np.all(self._state != -1, axis=1)
        self._state[is_complete] = -1

        indices, values = actions.T

        if np.any(self._state[self._arange, indices] != -1):
            raise RuntimeError('Invalid action: trying to set a variable '
                'that has already been assigned.')

        self._state[self._arange, indices] = values

        # Compute the rewards (more precisely, difference in log-rewards)
        rewards = np.zeros((self.num_envs,), dtype=np.float_)

        for clique, potential in self.potentials:
            # Check if the clique is active
            is_in_clique = np.any(clique == indices[:, None], axis=1)
            assignments = self._state[:, clique]
            full_assignment = np.all(assignments != -1, axis=1)
            is_active = np.logical_and(is_in_clique, full_assignment)

            # Get the codes for the assignments
            base = self.num_categories ** np.arange(len(clique))
            codes = np.sum(assignments[is_active] * base, axis=1)

            # Add the new potential
            rewards[is_active] += potential[codes]

        dones = np.all(self._state != -1, axis=1)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)

        return (self.observations(), rewards, dones, truncated, {})

    def observations(self):
        return {
            'variables': np.copy(self._state),
            'mask': (self._state == -1).astype(np.int_)
        }

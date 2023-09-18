import numpy as np
import gym
import math

from gym.spaces import Dict, Box, Discrete, MultiBinary

from gfn_maxent_rl.envs.treesample.policy import uniform_log_policy, action_mask


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

        observation_space = Dict({
            'variables': Box(
                low=-1,
                high=self.num_categories,
                shape=(self.num_variables,),
                dtype=np.int_
            ),
            'mask': MultiBinary(self.num_variables),
        })
        action_space = Discrete(self.num_variables * self.num_categories + 1)
        super().__init__(num_envs, observation_space, action_space)

    def reset(self, *, seed=None, options=None):
        self._state[:] = -1
        return (self.observations(), {})

    def step(self, actions):
        indices, values = divmod(actions, self.num_categories)
        dones = (indices == self.num_variables)

        if np.any(self._state[dones] == -1):
            raise RuntimeError('Invalid action: calling the stop action even '
                'though some variables have not been assigned.')
        
        indices, values = indices[~dones], values[~dones]

        if np.any(self._state[~dones, indices] != -1):
            raise RuntimeError('Invalid action: trying to set a variable '
                'that has already been assigned.')

        self._state[~dones, indices] = values

        # Compute the rewards (more precisely, difference in log-rewards)
        rewards = np.zeros((self.num_envs,), dtype=np.float_)

        for clique, potential in self.potentials:
            # Check if the clique is active
            is_in_clique = np.any(clique == indices[:, None], axis=1)
            assignments = self._state[:, clique][~dones]
            full_assignment = np.all(assignments != -1, axis=1)
            is_active = np.logical_and(is_in_clique, full_assignment)

            # Get the codes for the assignments
            base = self.num_categories ** np.arange(len(clique))
            codes = np.sum(assignments[is_active] * base, axis=1)

            # Add the new potential
            rewards[is_active] += potential[codes]

        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        self._state[dones] = -1  # Clear state for complete trajectories
        rewards[dones] = 0.  # Terminal action has 0 reward

        return (self.observations(), rewards, dones, truncated, {})

    def observations(self):
        return {
            'variables': np.copy(self._state),
            'mask': (self._state == -1).astype(np.int_)
        }

    # Properties & methods to interact with the replay buffer

    @property
    def observation_dtype(self):
        return np.dtype([
            ('variables', np.int_, (self.num_variables,)),
            ('mask', np.int_, (self.num_variables,))
        ])

    @property
    def max_length(self):
        return self.num_variables + 1

    def encode(self, observations):
        batch_size = observations['variables'].shape[0]
        encoded = np.empty((batch_size,), dtype=self.observation_dtype)
        encoded['variables'] = observations['variables']
        encoded['mask'] = observations['mask']
        return encoded

    def decode(self, observations):
        return {
            'variables': observations['variables'].astype(np.int32),
            'mask': observations['mask'].astype(np.float32),
        }

    def decode_sequence(self, samples):
        return self.decode(samples['observations'])

    # Method to interact with the algorithm (uniform sampling of action)

    def uniform_log_policy(self, observations):
        return uniform_log_policy(observations['mask'], self.num_categories)

    def num_parents(self, observations):
        return (observations['variables'] != -1).sum(axis=-1)

    def action_mask(self, observations):
        return action_mask(observations['mask'], self.num_categories)

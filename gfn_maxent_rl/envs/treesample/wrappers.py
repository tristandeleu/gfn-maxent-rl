import numpy as np
import gym
import jax.numpy as jnp
import math

from gfn_maxent_rl.envs.treesample.factor_graph_env import FactorGraphEnvironment


class FixedOrderingWrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env, FactorGraphEnvironment)
        super().__init__(env)
        self._permutation = np.append(self.env.permutation, -1)

    def observations(self):
        # The number of steps from the initial state
        steps = np.sum(self.env._state != -1, axis=1)
        # The index of the variable from the "permutation" array. If this is a
        # terminating transition (i.e., all the variables are set), then return -1
        variables = self._permutation[steps]
        # Repeated indices, e.g. [0, 0, 1, 1, 2, 2, 3, 3]
        indices = np.repeat(np.arange(self.env.num_variables), self.env.num_categories)

        return {
            'variables': np.copy(self.env._state),
            'mask': (indices == variables[:, None]).astype(np.int_)
        }

    def num_parents(self, observations):
        return jnp.ones((observations.shape[0],), dtype=jnp.int32)


class RewardCorrection(gym.Wrapper):
    def __init__(self, env, alpha=1., weight='all_steps'):
        super().__init__(env)
        self.alpha = alpha  # Temperature parameter
        self.weight = weight

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)

        if self.weight == 'all_steps':
            # num_steps = number of steps to the *next observation*
            num_steps = np.sum(observations['variables'] != -1, axis=1)
            correction = np.where(terminated | truncated, 0., -np.log(num_steps))

        elif self.weight == 'nonzero':
            if 'active_potentials' not in infos:
                raise KeyError('Unavaiable key `active_potentials` in `infos` dict.')

            weight = np.mean(infos['active_potentials'], axis=1)
            correction = np.where(terminated | truncated | (rewards == 0.),
                0., -math.lgamma(self.env.num_variables + 1) * weight)

        elif self.weight == 'uniform':
            total_correction = -math.lgamma(self.env.num_variables + 1)
            correction = np.where(terminated | truncated,
                0., total_correction / self.env.num_variables)

        else:
            raise ValueError(f'Unknown weight: {self.weight}')

        rewards = rewards + self.alpha * correction
        infos['correction'] = correction
        return (observations, rewards, terminated, truncated, infos)

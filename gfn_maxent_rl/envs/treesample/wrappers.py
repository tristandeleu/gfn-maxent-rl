import numpy as np
import gym
import jax.numpy as jnp
import math

from gym.spaces import MultiDiscrete

from gfn_maxent_rl.envs.treesample.factor_graph_env import FactorGraphEnvironment


class FixedOrderingWrapper(gym.Wrapper):
    def __init__(self, env):
        assert isinstance(env, FactorGraphEnvironment)
        super().__init__(env)

        self.action_space = MultiDiscrete([env.num_categories] * env.num_envs)
        self._index = np.zeros((env.num_envs,), dtype=np.int_)

    def reset(self, *, seed=None, options=None):
        self._index[:] = 0
        return super().reset(seed=seed, options=options)

    def step(self, values):
        variables = self.env.permutation[self._index]
        actions = np.vstack([variables, values]).T
        self._index = (self._index + 1) % self.env.num_variables
        return super().step(actions)

    # Method to interact with the algorithm (uniform sampling of action)

    def uniform_log_policy(self, observations):
        masks = self.action_mask(observations)
        return jnp.where(masks, -math.log(self.num_categories), -jnp.inf)

    def num_parents(self, observations):
        batch_size = observations.shape[0]
        return jnp.ones((batch_size,), dtype=jnp.int32)

    def action_mask(self, observations):
        shape = (observations['mask'].shape[0], self.num_variables * self.num_categories)
        log_pi = jnp.zeros(shape, dtype=jnp.bool_)
        slice_ = slice(
            self._index * self.num_categories,
            (self._index + 1) * self.num_categories
        )
        log_pi = log_pi.at[:, slice_].set(True)
        return log_pi

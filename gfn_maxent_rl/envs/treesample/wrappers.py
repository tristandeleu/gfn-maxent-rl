import numpy as np
import gym

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

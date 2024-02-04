import numpy as np
import gym
import jax.numpy as jnp

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

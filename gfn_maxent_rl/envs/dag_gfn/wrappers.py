import jax.numpy as jnp
import chex

from jumanji import types, wrappers
from typing import Tuple

from gfn_maxent_rl.envs.dag_gfn.types import DAGState, DAGObservation


class RewardCorrection(wrappers.Wrapper):
    def __init__(self, env, alpha=1.):
        super().__init__(env)
        self.alpha = alpha  # Temperature parameter

    def step(self, state: DAGState, action: chex.Array) -> Tuple[DAGState, types.TimeStep[DAGObservation]]:
        num_edges = jnp.sum(state.adjacency, dtype=jnp.float32)  # t
        state, timestep = super().step(state, action)

        # Correct the reward by subtracting log(t + 1), where t is the number
        # of edges in the current graph
        correction = jnp.where(timestep.mid(), -jnp.log1p(num_edges), 0.)
        timestep = timestep.replace(reward=timestep.reward + self.alpha * correction)

        return (state, timestep)

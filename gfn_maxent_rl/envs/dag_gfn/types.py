import jax.numpy as jnp
import chex

from typing import Optional


@chex.dataclass
class DAGState:
    adjacency: chex.Array
    closure_T: chex.Array
    key: chex.Array

    @classmethod
    def init(cls, num_nodes, key):
        adjacency = jnp.zeros((num_nodes, num_nodes), dtype=jnp.bool_)
        closure_T = jnp.eye(num_nodes, dtype=jnp.bool_)
        return cls(adjacency=adjacency, closure_T=closure_T, key=key)

    def update(self, source, target):
        adjacency = self.adjacency.at[source, target].set(True)
        closure_T = jnp.logical_or(
            self.closure_T,
            jnp.outer(self.closure_T[:, target], self.closure_T[source, :])
        )
        return DAGState(adjacency=adjacency, closure_T=closure_T, key=self.key)


@chex.dataclass
class DAGObservation:
    adjacency: chex.Array
    mask: chex.Array
    graph: Optional[chex.Array] = None

    @classmethod
    def from_state(cls, state):
        return cls(
            adjacency=state.adjacency.astype(jnp.float32),
            mask=1. - (state.adjacency + state.closure_T),
            graph=None
        )

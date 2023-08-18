import jax.numpy as jnp

from abc import ABC, abstractmethod


class GraphPrior(ABC):
    def __init__(self, num_variables):
        self.num_variables = num_variables

    @abstractmethod
    def log_prob(self, adjacency):
        """Computes log P(G).
        
        Parameters
        ----------
        adjacency : jnp.ndarray, shape `(num_variables, num_variables)`
            The adjacency matrix of the graph G.

        Returns
        -------
        log_prob : jnp.ndarray, shape `()`
            The log-prior of the graph G: log P(G).
        """
        pass

    def delta_score(self, adjacency, source, target):
        """Computes log P(G') - log P(G), where G' is the result of adding the
        edge X_i -> X_j to G.

        Parameters
        ----------
        adjacency : jnp.Array, shape `(num_variables, num_variables)`
            The adjacency matrix of the graph G.

        source : jnp.Array, shape `()`
            The index of the source of the edge to be added to G (X_i).

        target : jnp.Array, shape `()`
            The index of the target of the edge to be added to G (X_j)

        Returns
        -------
        delta_score : jnp.Array, shape `()`
            The difference in log priors log P(G') - log P(G).
        """
        next_adjacency = adjacency.at[source, target].set(True)
        return self.log_prob(next_adjacency) - self.log_prob(adjacency)

    @staticmethod
    def num_parents(adjacency):
        return jnp.count_nonzero(adjacency, axis=0)


class UniformPrior(GraphPrior):
    def __init__(self, num_variables):
        super().__init__(num_variables)
        self._log_prior = jnp.zeros((num_variables,), dtype=jnp.float32)

    def log_prob(self, adjacency):
        num_parents = UniformPrior.num_parents(adjacency)
        return jnp.sum(self._log_prior[num_parents])

    def delta_score(self, adjacency, source, target):
        return jnp.zeros(())

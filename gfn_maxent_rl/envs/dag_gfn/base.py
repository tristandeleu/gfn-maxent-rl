import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GraphPrior(ABC):
    num_variables: int

    @abstractmethod
    def log_prob(self, adjacencies):
        """Computes log P(G).
        
        Parameters
        ----------
        adjacencies : np.ndarray, shape `(num_graphs, num_variables, num_variables)`
            The adjacency matrix of the graph G.

        Returns
        -------
        log_prob : np.ndarray, shape `(num_graphs,)`
            The log-prior of the graph G: log P(G).
        """
        pass

    def delta_score(self, adjacencies, sources, targets):
        """Computes log P(G') - log P(G), where G' is the result of adding the
        edge X_i -> X_j to G.

        Parameters
        ----------
        adjacencies : np.ndarray, shape `(num_graphs, num_variables, num_variables)`
            The adjacency matrix of the graph G.

        sources : np.ndarray, shape `(num_graphs,)`
            The index of the source of the edge to be added to G (X_i).

        targets : np.ndarray, shape `(num_graphs,)`
            The index of the target of the edge to be added to G (X_j)

        Returns
        -------
        delta_score : np.ndarray, shape `(num_graphs,)`
            The difference in log priors log P(G') - log P(G).
        """
        num_graphs = adjacencies.shape[0]
        next_adjacencies = np.copy(adjacencies)
        next_adjacencies[np.arange(num_graphs), sources, targets] = True
        return self.log_prob(next_adjacencies) - self.log_prob(adjacencies)

    @staticmethod
    def num_parents(adjacencies):
        return np.count_nonzero(adjacencies, axis=2)


@dataclass
class MarginalLikelihood(ABC):
    data: np.ndarray

    @abstractmethod
    def local_score(self, variables, parents):
        """Computes the local score LocalScore(X_j | Pa_G(X_j)).

        Parameters
        ----------
        variables : np.ndarray, shape `(num_graphs,)`
            The variable X_j to compute the local-score of.

        parents : np.ndarray, shape `(num_graphs, num_variables)`
            The binary mask representing the parents Pa_G(X_j) of X_j in G.
            This corresponds to the j'th column of the adjacency matrix of G.

        Returns
        -------
        local_score : np.ndarray, shape `(num_graphs,)`
            The local score LocalScore(X_j | Pa_G(X_j)).
        """
        pass

    def delta_score(self, adjacencies, sources, targets):
        """Computes the delta-score for adding an edge X_i -> X_j to some grpah
        G, for a specific choice of local score. The delta-score is given by:

            LocalScore(X_j | Pa_G(X_j) U X_i) - LocalScore(X_j | Pa_G(X_j))
        
        Parameters
        ----------
        adjacencies : np.ndarray, shape `(num_graphs, num_variables, num_variables)`
            The adjacency matrix of the graph G.

        sources : np.ndarray, shape `(num_graphs,)`
            The index of the source of the edge to be added to G (X_i).

        targets : np.ndarray, shape `(num_graphs,)`
            The index of the target of the edge to be added to G (X_j)

        Returns
        -------
        delta_score : jnp.Array, shape `(num_graphs,)`
            The delta-score for adding the edge X_i -> X_j to the graph G.
        """
        arange = np.arange(adjacencies.shape[0])
        parents = adjacencies[arange, :, targets]
        next_parents = np.copy(parents)
        next_parents[arange, sources] = True
        return (
            self.local_score(targets, next_parents)
            - self.local_score(targets, parents)
        )

    def log_prob(self, adjacencies):
        """Compute the log-marginal likelihood for a graph G:

            log P(D | G) = \sum_j LocalScore(X_j | Pa_G(X_j))
        
        Parameters
        ----------
        adjacencies : np.ndarray, shape `(num_graphs, num_variables, num_variables)`
            The adjacency matrix of the graph G.

        Returns
        -------
        log_prob : np.ndarray, shape `(num_graphs,)`
            The log-marginal likelihood log P(D | G).
        """
        num_graphs, num_variables = adjacencies.shape[:2]
        adjacencies = adjacencies.transpose(0, 2, 1)
        parents = adjacencies.reshape(-1, num_variables)
        variables = np.tile(np.arange(num_variables), num_graphs)

        local_scores = self.local_score(variables, parents)
        local_scores = local_scores.reshape(num_graphs, num_variables)
        return np.sum(local_scores, axis=1)


@dataclass
class JointModel:
    graph_prior: GraphPrior
    marginal_likelihood: MarginalLikelihood

    def log_prob(self, adjacencies):
        return (
            self.graph_prior.log_prob(adjacencies)
            + self.marginal_likelihood.log_prob(adjacencies)
        )

    def delta_score(self, adjacencies, sources, targets):
        return (
            self.graph_prior.delta_score(adjacencies, sources, targets)
            + self.marginal_likelihood.delta_score(adjacencies, sources, targets)
        )

    @property
    def num_variables(self):
        if self.marginal_likelihood.data.shape[1] != self.graph_prior.num_variables:
            raise ValueError('The data has a different number of columns compared '
                f'to `num_variables`: {self.marginal_likelihood.data.shape[1]} != '
                f'{self.graph_prior.num_variables}')
        return self.graph_prior.num_variables

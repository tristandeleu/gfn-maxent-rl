import jax.numpy as jnp
import jax
import math

from abc import ABC, abstractmethod
from jax.scipy.special import gammaln


class Scorer(ABC):
    @abstractmethod
    def local_score(self, variable, parents):
        """Computes the local score LocalScore(X_j | Pa_G(X_j)).

        Parameters
        ----------
        variable : jnp.Array, shape `()`
            The variable X_j to compute the local-score of.

        parents : jnp.Array, shape `(num_variables,)`
            The binary mask representing the parents Pa_G(X_j) of X_j in G.
            This corresponds to the j'th column of the adjacency matrix of G.

        Returns
        -------
        local_score : jnp.Array, shape `()`
            The local score LocalScore(X_j | Pa_G(X_j)).
        """
        pass

    def delta_score(self, adjacency, source, target):
        """Computes the delta-score for adding an edge X_i -> X_j to some grpah
        G, for a specific choice of local score. The delta-score is given by:

            LocalScore(X_j | Pa_G(X_j) U X_i) - LocalScore(X_j | Pa_G(X_j))
        
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
            The delta-score for adding the edge X_i -> X_j to the graph G.
        """
        parents = adjacency[:, target]
        next_parents = parents.at[source].set(True)
        return (
            self.local_score(target, next_parents)
            - self.local_score(target, parents)
        )

    def log_prob(self, adjacency):
        """Compute the log-marginal likelihood for a graph G:

            log P(D | G) = \sum_j LocalScore(X_j | Pa_G(X_j))
        
        Parameters
        ----------
        adjacency : jnp.Array, shape `(num_variables, num_variables)`
            The adjacency matrix of the graph G.    

        Returns
        -------
        log_prob : jnp.Array, shape `()`
            The log-marginal likelihood log P(D | G).
        """
        num_variables = adjacency.shape[0]
        variables = jnp.arange(num_variables)
        scores = jax.vmap(self.local_score, in_axes=(1, 0))(variables, adjacency)
        return jnp.sum(scores)


class ZeroScorer(Scorer):
    def local_score(self, parents, variable):
        return jnp.zeros(())

    def delta_score(self, adjacency, source, target):
        return jnp.zeros(())


class LinearGaussianScorer(Scorer):
    def __init__(self, data, prior_mean=0., prior_scale=1., obs_scale=math.sqrt(0.1)):
        self.data = data
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.obs_scale = obs_scale

    def local_score(self, variable, parents):
        # https://tristandeleu.notion.site/Linear-Gaussian-Score-16a2ed3422fb4f1fa0b3f554ff57f67d
        num_samples, num_variables = self.data.shape
        masked_data = self.data * parents

        mean = self.prior_mean * jnp.sum(masked_data, axis=1)
        diff = (self.data[variable] - mean) / self.obs_scale
        Y = self.prior_scale * jnp.dot(masked_data.T, diff)
        Sigma = (
            (self.obs_scale ** 2) * jnp.eye(num_variables)
            + (self.prior_scale ** 2) * jnp.matmul(masked_data.T, masked_data)
        )

        term1 = jnp.sum(diff ** 2)
        term2 = -jnp.vdot(Y, jnp.linalg.solve(Sigma, Y))
        _, term3 = jnp.linalg.slogdet(Sigma)
        term4 = 2 * (num_samples - num_variables) * math.log(self.obs_scale)
        term5 = num_samples * math.log(2 * math.pi)

        return -0.5 * (term1 + term2 + term3 + term4 + term5)


class BGeScorer(Scorer):
    def __init__(self, data, mean_obs=None, alpha_mu=1., alpha_w=None):
        num_variables = data.shape[1]
        if mean_obs is None:
            mean_obs = jnp.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.

        self.data = data
        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples, self.num_variables = self.data.shape
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * jnp.eye(self.num_variables)
        data_mean = jnp.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + jnp.dot(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * jnp.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = jnp.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

    def local_score(self, variable, parents):
        def _logdet(array, mask):
            mask = jnp.outer(mask, mask)
            array = mask * array + (1. - mask) * jnp.eye(self.num_variables)
            _, logdet = jnp.linalg.slogdet(array)
            return logdet

        num_parents = jnp.sum(parents)
        parents_and_variable = parents.at[variable].set(True)
        factor = self.num_samples + self.alpha_w - self.num_variables + num_parents

        log_term_r = (
            0.5 * factor * _logdet(self.R, parents)
            - 0.5 * (factor + 1) * _logdet(self.R, parents_and_variable)
        )

        return self.log_gamma_term[num_parents] + log_term_r

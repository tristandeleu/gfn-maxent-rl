import numpy as np
import math

from scipy.special import gammaln

from gfn_maxent_rl.envs.dag_gfn.base import MarginalLikelihood


class ZeroScore(MarginalLikelihood):
    def local_score(self, variables, parents):
        return np.zeros((variables.shape[0],), dtype=np.float32)

    def delta_score(self, adjacencies, sources, targets):
        return np.zeros((adjacencies.shape[0],), dtype=np.float32)


class LinearGaussianScore(MarginalLikelihood):
    def __init__(self, data, prior_mean=0., prior_scale=1., obs_scale=math.sqrt(0.1)):
        super().__init__(data)
        self.prior_mean = prior_mean
        self.prior_scale = prior_scale
        self.obs_scale = obs_scale

    def local_score(self, variables, parents):
        num_samples, num_variables = self.data.shape
        masked_data = self.data * parents[:, None]

        means = self.prior_mean * np.sum(masked_data, axis=2)
        diffs = (self.data[:, variables].T - means) / self.obs_scale
        Y = self.prior_scale * np.matmul(diffs[:, None], masked_data)
        Y = np.squeeze(Y, axis=1)
        Sigma = (
            (self.obs_scale ** 2) * np.eye(num_variables)
            + (self.prior_scale ** 2) * np.matmul(masked_data.transpose(0, 2, 1), masked_data)
        )

        term1 = np.sum(diffs ** 2, axis=1)
        term2 = -np.sum(Y * np.linalg.solve(Sigma, Y), axis=1)
        _, term3 = np.linalg.slogdet(Sigma)
        term4 = 2 * (num_samples - num_variables) * math.log(self.obs_scale)
        term5 = num_samples * math.log(2 * math.pi)

        return -0.5 * (term1 + term2 + term3 + term4 + term5)


class BGeScore(MarginalLikelihood):
    def __init__(self, data, mean_obs=None, alpha_mu=1., alpha_w=None):
        super().__init__(data)
        num_variables = data.shape[1]
        if mean_obs is None:
            mean_obs = np.zeros((num_variables,))
        if alpha_w is None:
            alpha_w = num_variables + 2.

        self.mean_obs = mean_obs
        self.alpha_mu = alpha_mu
        self.alpha_w = alpha_w

        self.num_samples, self.num_variables = self.data.shape
        self.t = (self.alpha_mu * (self.alpha_w - self.num_variables - 1)) / (self.alpha_mu + 1)

        T = self.t * np.eye(self.num_variables)
        data_mean = np.mean(data, axis=0, keepdims=True)
        data_centered = data - data_mean

        self.R = (T + np.matmul(data_centered.T, data_centered)
            + ((self.num_samples * self.alpha_mu) / (self.num_samples + self.alpha_mu))
            * np.dot((data_mean - self.mean_obs).T, data_mean - self.mean_obs)
        )
        all_parents = np.arange(self.num_variables)
        self.log_gamma_term = (
            0.5 * (math.log(self.alpha_mu) - math.log(self.num_samples + self.alpha_mu))
            + gammaln(0.5 * (self.num_samples + self.alpha_w - self.num_variables + all_parents + 1))
            - gammaln(0.5 * (self.alpha_w - self.num_variables + all_parents + 1))
            - 0.5 * self.num_samples * math.log(math.pi)
            + 0.5 * (self.alpha_w - self.num_variables + 2 * all_parents + 1) * math.log(self.t)
        )

    def local_score(self, variables, parents):
        def _logdet(array, mask):
            mask = mask[:, None, :] * mask[:, :, None]
            array = mask * array + (1. - mask) * np.eye(self.num_variables)
            _, logdet = np.linalg.slogdet(array)
            return logdet

        num_parents = np.sum(parents, axis=1)  # (num_graphs,)
        arange = np.arange(parents.shape[0])
        parents_and_variable = np.copy(parents)  # (num_graphs, num_variables)
        parents_and_variable[arange, variables] = True

        factor = self.num_samples + self.alpha_w - self.num_variables + num_parents

        log_term_r = (
            0.5 * factor * _logdet(self.R, parents)
            - 0.5 * (factor + 1) * _logdet(self.R, parents_and_variable)
        )

        return self.log_gamma_term[num_parents] + log_term_r

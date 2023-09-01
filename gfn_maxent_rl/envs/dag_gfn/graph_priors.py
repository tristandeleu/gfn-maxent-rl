import numpy as np

from gfn_maxent_rl.envs.dag_gfn.base import GraphPrior


class UniformPrior(GraphPrior):
    def __init__(self, num_variables):
        super().__init__(num_variables)
        self._log_prior = np.zeros((num_variables,), dtype=np.float32)

    def log_prob(self, adjacencies):
        num_parents = UniformPrior.num_parents(adjacencies)
        return np.sum(self._log_prior[num_parents], axis=1)

    def delta_score(self, adjacencies, sources, targets):
        return np.zeros((adjacencies.shape[0],), dtype=np.float32)

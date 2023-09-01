import numpy as np

from numpy.random import default_rng

from gfn_maxent_rl.envs.dag_gfn.scores import ZeroScore, LinearGaussianScore


rng = default_rng(0)
data = rng.normal(size=(100, 3))  # Dummy data

score = LinearGaussianScore(data=data, prior_mean=0., prior_scale=1.)

parents = np.array([
    [True, False, False],
    [False, False, False],
    [False, True, True],
    [False, True, False]
])
variables = np.array([1, 2, 0, 2])

local_scores = score.local_score(variables, parents)
print(local_scores)
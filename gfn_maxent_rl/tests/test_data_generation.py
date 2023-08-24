from numpy.random import default_rng

from gfn_maxent_rl.envs.dag_gfn.data_generation.graphs import sample_erdos_renyi_graph, sample_linear_gaussian
from gfn_maxent_rl.envs.dag_gfn.data_generation.data import sample_from_linear_gaussian


rng = default_rng(0)

# Randomly generated Bayesian Network (structure & CPDs)
bn = sample_erdos_renyi_graph(num_variables=5, num_edges=5, rng=rng)
bn = sample_linear_gaussian(bn, rng=rng)

# Sample data from the Bayesian Network
dataset = sample_from_linear_gaussian(bn, num_samples=100, rng=rng)

print(dataset.head())

import numpy as np
import networkx as nx

from numpy.random import default_rng

from gfn_maxent_rl.envs.treesample.factor_graph_env import FactorGraphEnvironment


def chain_env(
    num_envs,
    num_variables,
    num_categories,
    rng=default_rng(),
    rbf_bandwidth=1.,
    rbf_scale=0.5,
    factor=2.5,
):
    # Create the covariance matrix
    x, y = np.arange(num_variables), np.arange(num_categories)
    X, Y = np.meshgrid(x, y)
    XY = np.dstack([X, Y]).reshape(-1, 2)
    dists = np.sum((XY[:, None] - XY) ** 2, axis=2)
    covariance = rbf_scale * np.exp(-0.5 * dists / rbf_bandwidth)

    # Sample unary potentials from a GP prior
    unary_potentials = rng.multivariate_normal(
        mean=np.zeros((covariance.shape[0],)),
        cov=covariance
    )
    unary_potentials = unary_potentials.reshape(num_variables, num_categories)

    # Create the graph
    graph = nx.path_graph(num_variables, create_using=nx.Graph)

    # Create potentials. Unary potentials
    potentials = []
    for variable, potential in enumerate(unary_potentials):
        potentials.append((np.array([variable]), potential))

    # Binary potentials
    values = np.arange(num_categories)
    dists = np.abs(values[:, None] - values)
    binary = factor * np.minimum(dists, num_categories - dists)
    X, Y = np.meshgrid(values, values)
    binary_potentials = np.zeros((num_categories ** 2,), dtype=np.float_)
    binary_potentials[X * num_categories + Y] = binary

    for edge in graph.edges:
        potentials.append((np.asarray(edge, dtype=np.int_), binary_potentials))

    permutation = np.arange(num_variables)

    return FactorGraphEnvironment(num_envs, graph, potentials, permutation)

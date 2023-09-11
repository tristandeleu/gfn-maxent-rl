import numpy as np
import networkx as nx

from numpy.random import default_rng

from gfn_maxent_rl.envs.treesample.factor_graph_env import FactorGraphEnvironment


def permuted_chain_env(
    num_envs,
    num_variables,
    num_categories,
    rng=default_rng(),
    alpha=1.,
):
    # Generate binary factors from symmetric Dirichlet
    alphas = np.full((num_categories,), alpha)
    cond_distributions = rng.dirichlet(alphas, size=(num_variables, num_categories))

    # Create the graph (permuted chain)
    graph = nx.path_graph(num_variables, create_using=nx.Graph)
    mapping = dict(enumerate(rng.permutation(num_variables)))
    graph = nx.relabel_nodes(graph, mapping, copy=True)

    # Assign the binary potentials
    potentials = []
    for (source, target) in graph.edges:
        values = np.arange(num_categories)
        X, Y = np.meshgrid(values, values)

        potential = np.zeros((num_categories ** 2,), dtype=np.float_)
        potential[X * num_categories + Y] = cond_distributions[target]
        potentials.append((np.asarray([source, target], dtype=np.int_), potential))
    
    permutation = np.arange(num_variables)

    return FactorGraphEnvironment(num_envs, graph, potentials, permutation)

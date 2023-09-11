import numpy as np
import networkx as nx
import math

from numpy.random import default_rng

from gfn_maxent_rl.envs.treesample.factor_graph_env import FactorGraphEnvironment
from gfn_maxent_rl.envs.dag_gfn.data_generation.graphs import sample_erdos_renyi_graph


def factor_graph1_env(
    num_envs,
    num_variables,
    num_categories,
    rng=default_rng(),
    max_retries=1000,
    max_clique_size=4,
):
    # Create a random graph (Erdos-Renyi) with a single connected component
    for _ in range(max_retries):
        graph = sample_erdos_renyi_graph(
            num_variables=num_variables,
            p=2 * math.log(num_variables) / num_variables,
            nodes=np.arange(num_variables),
            create_using=nx.Graph,
            rng=rng
        )
        if nx.number_connected_components(graph) == 1:
            # Ensure that there is no clique of size > max_clique_size
            cliques = list(nx.find_cliques(graph))
            if all(len(clique) <= max_clique_size for clique in cliques):
                break
    else:
        raise RuntimeError('Unable to create a random graph over '
            f'{num_variables} variables with a single connected component, '
            f'or with all cliques of size < {max_clique_size}.')

    # Randomly generate the factors (potentials samples from normal distribution)
    potentials = []
    for clique in cliques:
        num_states = num_categories ** len(clique)
        potentials.append((
            np.asarray(clique, dtype=np.int_),
            rng.normal(loc=0., scale=1., size=(num_states,))
        ))
    
    # Find the variable ordering based on the heuristic
    permutation = []
    for clique in sorted(cliques, key=len, reverse=True):
        for variable in clique:
            if variable not in permutation:
                permutation.append(variable)
    assert len(permutation) == num_variables
    permutation = np.asarray(permutation, dtype=np.int_)

    return FactorGraphEnvironment(num_envs, graph, potentials, permutation)

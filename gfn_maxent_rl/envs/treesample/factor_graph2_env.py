import numpy as np
import networkx as nx
import math
import warnings

from numpy.random import default_rng
from itertools import chain, product

from gfn_maxent_rl.envs.treesample.factor_graph_env import FactorGraphEnvironment
from gfn_maxent_rl.envs.dag_gfn.data_generation.graphs import sample_erdos_renyi_graph


def factor_graph2_env(
    num_envs,
    num_variables,
    num_categories,
    rng=default_rng(),
    max_retries=1000,
    max_clique_size=4,
    factor=2.,
):
    assert num_variables % 2 == 0

    if num_categories != 2:
        warnings.warn('The environment `factor_graph2` only accepts binary '
            'variables, but was instantiated with '
            f'`num_categories={num_categories}`. Ignoring this argument, '
            'and using binary variables instead.', stacklevel=2)

    # Create a random graph (Erdos-Renyi) with a single connected component
    for _ in range(max_retries):
        graph = sample_erdos_renyi_graph(
            num_variables=num_variables // 2,
            p=3 * math.log(num_variables // 2) / num_variables,
            nodes=np.arange(num_variables // 2),
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
        clique_size = len(clique)
        # Connect factor to either X_2n+1 or X_2n+2
        offset = rng.integers(2, size=(clique_size,))

        # Get all the states
        states = np.fromiter(
            chain(*product([0, 1], repeat=clique_size)),
            dtype=np.int_,
            count=clique_size * (1 << clique_size)
        ).reshape(-1, clique_size)

        potentials.append((
            2 * np.asarray(clique, dtype=np.int_) + offset,
            factor * (np.sum(states, axis=1) >= (0.5 * clique_size))
        ))
    
    # Add XOR binary factor
    xor_potential = np.array([0., 1., 1., 0.])
    for n in range(0, num_variables, 2):
        potentials.append((
            np.array([n, n + 1], dtype=np.int_),
            factor * xor_potential
        ))
    
    # Find the variable ordering based on the heuristic
    permutation = []
    cliques = [clique for (clique, _) in potentials]
    for clique in sorted(cliques, key=len, reverse=True):
        for variable in clique:
            if variable not in permutation:
                permutation.append(variable)
    assert len(permutation) == num_variables
    permutation = np.asarray(permutation, dtype=np.int_)

    return FactorGraphEnvironment(num_envs, graph, potentials, permutation)

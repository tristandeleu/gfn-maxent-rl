import numpy as np
import networkx as nx
import string
import math

from numpy.random import default_rng
from itertools import chain, product, islice, count


def sample_erdos_renyi_graph(
        num_variables,
        p=None,
        num_edges_per_node=None,
        nodes=None,
        create_using=nx.DiGraph,
        rng=default_rng()
    ):
    if p is None:
        if num_edges_per_node is None:
            raise ValueError('One of p or num_edges must be specified.')
        p = 2. * num_edges_per_node / (num_variables - 1)
    
    if nodes is None:
        uppercase = string.ascii_uppercase
        iterator = chain.from_iterable(
            product(uppercase, repeat=r) for r in count(1))
        nodes = [''.join(letters) for letters in islice(iterator, num_variables)]

    adjacency = rng.binomial(1, p=p, size=(num_variables, num_variables))
    adjacency = np.tril(adjacency, k=-1)  # Only keep the lower triangular part

    # Permute the rows and columns
    perm = rng.permutation(num_variables)
    adjacency = adjacency[perm, :]
    adjacency = adjacency[:, perm]

    graph = nx.from_numpy_array(adjacency, create_using=create_using)
    mapping = dict(enumerate(nodes))
    nx.relabel_nodes(graph, mapping=mapping, copy=False)

    return graph


def sample_linear_gaussian(
        graph,
        loc_edges=0.0,
        scale_edges=1.0,
        obs_scale=math.sqrt(0.1),
        rng=default_rng()
    ):
    # Set graph attribute to identify Linear Gaussian Bayesian Network
    graph.graph['type'] = 'linear-gaussian'

    # Create the model parameters
    attrs = {}
    for node in graph.nodes:
        parents = list(graph.predecessors(node))

        # Sample random parameters of the CPDs (from Normal distribution).
        theta = rng.normal(loc_edges, scale_edges, size=(len(parents),))

        # Add CPD & observation scale
        attrs[node] = {
            'parents': parents,
            'cpd': theta,
            'bias': 0.,  # No bias term by default
            'obs_scale': obs_scale
        }
    nx.set_node_attributes(graph, attrs)

    return graph

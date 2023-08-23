import numpy as np
import pandas as pd
import networkx as nx

from numpy.random import default_rng


def sample_from_linear_gaussian(graph, num_samples, rng=default_rng()):
    """Sample from a linear-Gaussian model using ancestral sampling."""
    if graph.graph.get('type', '') != 'linear-gaussian':
        raise ValueError('The graph is not a Linear Gaussian Bayesian Network.')

    samples = pd.DataFrame(columns=list(graph.nodes()))
    for node in nx.topological_sort(graph):
        attrs = graph.node[node]
        if attrs['parents']:
            values = np.vstack([samples[parent] for parent in attrs['parents']])
            mean = attrs['bias'] + np.dot(attrs['cpd'], values)
            samples[node] = rng.normal(mean, attrs['obs_scale'])
        else:
            samples[node] = rng.normal(
                attrs['bias'],
                attrs['obs_scale'],
                size=(num_samples,)
            )
    return samples
